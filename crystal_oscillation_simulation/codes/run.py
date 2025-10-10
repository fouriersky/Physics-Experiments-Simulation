import numpy as np
from opt_method import relax_positions_fixed_cell
from MD_module import run_md, virial_stress_full

def apply_strain(lattice, strain_tensor):
    """
    对晶胞施加小应变: new_lattice = (I + ε) * lattice
    lattice: 3x3 原始晶胞矩阵
    strain_tensor: 3x3 对称应变张量
    """
    deformation = np.eye(3) + strain_tensor
    return deformation @ lattice

def stress_strain_curve(atoms, lattice, strain_list, T=0.0, steps=1000, dt=1e-3):
    """
    计算应力-应变关系
    atoms: 原子坐标 (N, 3)
    lattice: 晶胞矩阵 (3, 3)
    strain_list: 应变大小列表，例如 [0.0, 0.005, 0.01, -0.005, -0.01]
    T: 温度 (K)，T=0 相当于结构优化后能量极小点
    steps, dt: MD 步数和时间步长
    """
    results = []

    for eps in strain_list:
        # 构造一个 xx 单轴拉伸应变
        strain_tensor = np.array([[eps, 0, 0],
                                  [0,   0, 0],
                                  [0,   0, 0]])
        
        new_lattice = apply_strain(lattice, strain_tensor)
        
        # ========== 0K 情况 ==========
        if T == 0:
            atoms_opt, new_lat_opt = relax_positions_fixed_cell(atoms, new_lattice)
            stress = virial_stress_full(atoms_opt, new_lat_opt)
        
        # ========== 有限温度 ==========
        else:
            traj = run_md(atoms, new_lattice, T=T, steps=steps, dt=dt)
            stress = virial_stress_full(traj[-1], new_lattice)  # 用末态近似，也可以做时间平均
        
        results.append((eps, stress))

    return results


if __name__ == "__main__":
    # ====== 示例：初始化晶格和原子 ======
    # fcc Al 初始晶格参数 a0 = 4.05 Å
    a0 = 4.05
    lattice = np.array([[a0, 0, 0],
                        [0, a0, 0],
                        [0, 0, a0]])
    # 一个简单的 1 原子单胞
    atoms = np.array([[0.0, 0.0, 0.0]])

    # 定义应变范围
    strain_list = np.linspace(-0.02, 0.02, 5)  # -2% 到 +2%
    
    # 计算应力-应变关系
    results = stress_strain_curve(atoms, lattice, strain_list, T=0.0)

    # 输出
    print("应变-应力结果 (xx 分量):")
    for eps, stress in results:
        print(f"strain = {eps:.4f}, stress_xx = {stress[0,0]:.6f}")
