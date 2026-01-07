import numpy as np
from SK import SKParameters, SKTightBinding, get_high_symmetry_path

def apply_strain(model: SKTightBinding, epsilon: float, axis: str = 'x', poisson_ratio: float = 0.0):
    """
    对 SKTightBinding 模型施加单轴应变。
    直接修改 model.lattice 中的实空间矢量和倒易矢量。
    
    参数:
        model: SKTightBinding 实例
        epsilon: 应变量 (例如 0.01 表示拉伸 1%, -0.01 表示压缩 1%)
        axis: 施加应变的方向 'x', 'y', 或 'z'
        poisson_ratio: 泊松比 (默认 0.0，即不考虑横向收缩；Si 约为 0.28)
    """
    print(f"--- Applying Strain: {epsilon*100:.2f}% along {axis}-axis (Poisson={poisson_ratio}) ---")
    
    # 1. 构建应变张量 (Strain Tensor)
    # e.g. axis='x' -> [[1+eps, 0, 0], [0, 1-nu*eps, 0], [0, 0, 1-nu*eps]]
    strain_tensor = np.eye(3)
    transverse_scale = 1.0 - poisson_ratio * epsilon
    axial_scale = 1.0 + epsilon
    
    if axis == 'x':
        strain_tensor[0, 0] = axial_scale
        strain_tensor[1, 1] = transverse_scale
        strain_tensor[2, 2] = transverse_scale
    elif axis == 'y':
        strain_tensor[0, 0] = transverse_scale
        strain_tensor[1, 1] = axial_scale
        strain_tensor[2, 2] = transverse_scale
    elif axis == 'z':
        strain_tensor[0, 0] = transverse_scale
        strain_tensor[1, 1] = transverse_scale
        strain_tensor[2, 2] = axial_scale
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    lat = model.lattice

    # 2. 变形原胞基矢 (Primitive Vectors)
    # a_new = a_old @ strain_tensor
    lat.a1 = np.dot(lat.a1, strain_tensor)
    lat.a2 = np.dot(lat.a2, strain_tensor)
    lat.a3 = np.dot(lat.a3, strain_tensor)

    # 3. 变形最近邻位移向量 (Nearest Neighbor Vectors)
    # 这会直接改变键长和键角，进而通过 SK 缩放定律改变 hopping 参数
    new_nn_R = []
    for R in lat.nn_R:
        new_nn_R.append(np.dot(R, strain_tensor))
    lat.nn_R = np.array(new_nn_R)

    # 4. 重新计算倒易格矢 (Reciprocal Vectors)
    # 因为实空间晶胞变形了，布里渊区也会变形
    V = np.dot(lat.a1, np.cross(lat.a2, lat.a3))
    lat.b1 = 2 * np.pi * np.cross(lat.a2, lat.a3) / V
    lat.b2 = 2 * np.pi * np.cross(lat.a3, lat.a1) / V
    lat.b3 = 2 * np.pi * np.cross(lat.a1, lat.a2) / V
    
    # 打印变形后的键长信息
    d_new = np.linalg.norm(lat.nn_R, axis=1)
    print(f"Deformed bond lengths (Å): {d_new}")
    print(f"Mean bond length: {np.mean(d_new):.4f} Å")

def debug_check_scaling(model: SKTightBinding):
    R0 = model.lattice.nn_R[0]
    l,m,n,d = model.lattice.direction_cosines(R0)
    V0 = model.params.scaled_params_at_distance(d)
    print(f"First bond length d = {d:.6f} Å")
    print("Scaled two-center (subset):",
          {k: round(V0[k],6) for k in ['sss','sps','pps','ppp']})

if __name__ == "__main__":
    # --- 1. 初始化模型 ---
    params = SKParameters()
    # 可以在这里微调参数以获得更好的带隙，如果需要的话
    # params.E_s1 += 2.0 
    
    model = SKTightBinding(params, basis="sp3d5s1")
    model.distance_for_scaling = None

    #print("[Before strain]")
    #debug_check_scaling(model)
    #apply_strain(model, epsilon=0.02, axis='z', poisson_ratio=0.28)
    #print("[After strain]")
    #debug_check_scaling(model)

    
    # --- 2. 施加应变 ---
    # 沿 Z 轴拉伸 2% (模拟外延生长或机械拉伸)
    # 泊松比设为 0.28 (Si 的典型值)，意味着 Z 拉伸时 X,Y 会收缩
    apply_strain(model, epsilon=0.02, axis='z', poisson_ratio=0.28)

    # --- 3. 计算能带结构 ---
    # 注意：应变后高对称点的位置在倒易空间也会发生变化
    # 这里我们使用分数坐标定义路径，get_high_symmetry_path 会自动利用
    # 变形后的倒易基矢 (b1, b2, b3) 计算正确的笛卡尔 k 点
    points = {
        'Γ': np.array([0.0, 0.0, 0.0]),
        'X': np.array([0.5, 0.0, 0.5]), 
        'L': np.array([0.5, 0.5, 0.5]),
        'K': np.array([0.375, 0.375, 0.75]),
    }
    k_path, labels, pos = get_high_symmetry_path(
        a=params.a, # 这里的 a 仅用于初始化 helper，实际计算用的是 model 内部变形后的 lattice
        points=points,
        segments=[('L', 'Γ'), ('Γ', 'X'), ('X', 'K'), ('K', 'Γ')],
        n_points=60
    )
    print("Plotting strained band structure...")
    model.plot_band_structure(
        k_path, 
        k_labels=labels, 
        k_positions=pos,
        electrons_per_cell=8, 
        spin_degeneracy=2, 
        shift_mode="VBM", 
        ymin=-4, ymax=4,
        save_path='./pic/nonstrained_bands.png',
        show_gap=True
    )
    
    # --- 4. (可选) 简单的各向异性检查 ---
    # 检查 Γ 点附近的本征值分裂
    H_gamma = model.hamiltonian_k(np.array([0,0,0]))
    evals_g = np.linalg.eigvalsh(H_gamma)
    # 排序并查看 VBM 附近的简并度
    evals_g.sort()
    occ_idx = 4 # sp3d5s1, 8 electrons -> 4 bands filled
    vbm_states = evals_g[occ_idx-3 : occ_idx] # 查看价带顶的几个态
    print(f"Eigenvalues at Gamma near VBM: {vbm_states}")
    print("If strain is effective, you should see splitting in the top valence bands.")
    