import numpy as np
from scipy.ndimage import map_coordinates

# ---------- 1) 读取数据 ----------
npz_path = "./225_X_5.npz"
data = np.load(npz_path)
# 形状: (n_bands, 3, Nx, Ny, Nz)，这里 n_bands=3, Nx=Ny=Nz=4
H_stack = data["H"]  # (3, 3, 4, 4, 4) 若保存时为 (n_bands,3,4,4,4)

# 兼容可能的两种维度顺序
if H_stack.ndim == 5 and H_stack.shape[1] == 3:
    # (n_bands, 3, Nx, Ny, Nz)
    pass
elif H_stack.ndim == 4 and H_stack.shape[0] == 3:
    # 单带 (3, Nx, Ny, Nz) -> (1, 3, Nx, Ny, Nz)
    H_stack = H_stack[None, ...]
else:
    # 若为 (3, n_bands, Nx, Ny, Nz) 则转置
    if H_stack.ndim == 5 and H_stack.shape[0] == 3:
        H_stack = np.moveaxis(H_stack, 1, 0)


n_bands, _, Nx, Ny, Nz = H_stack.shape
grid = {
    "Nx": Nx, "Ny": Ny, "Nz": Nz,
    "x": np.linspace(0.0, 1.0, Nx, endpoint=False),
    "y": np.linspace(0.0, 1.0, Ny, endpoint=False),
    "z": np.linspace(0.0, 1.0, Nz, endpoint=False),
}

# ---------- 2) FCC 斜基：与 MPB geometry_lattice 一致 ----------
A = np.eye(3)

# 使用以晶胞中心为原点的分数坐标：[-0.5, 0.5)
x = (np.arange(Nx) / Nx) - 0.5
y = (np.arange(Ny) / Ny) - 0.5
z = (np.arange(Nz) / Nz) - 0.5
grid = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "x": x, "y": y, "z": z}

def rotation_matrix_cart(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle); s = np.sin(angle); C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
    ])
    return R

def op_from_cart(name, R_cart, t_cart=(0.0, 0.0, 0.0)):
    Ainv = np.linalg.inv(A)
    # 仍保留 R_frac（可用于检查/日志），但实际坐标变换用 R_cart/t_cart
    R_frac = Ainv @ R_cart @ A
    R_frac_rounded = np.round(R_frac)
    if np.linalg.norm(R_frac - R_frac_rounded) < 1e-10:
        R_frac = R_frac_rounded
    t_frac = Ainv @ np.asarray(t_cart, dtype=float)
    return {"name": name, "R_frac": R_frac, "R_cart": R_cart, "t_frac": t_frac}

k_frac_X = np.array([0.0, 0.0, 0.5], dtype=float)
k_cart_X = k_frac_X @ A.T 

def d4h_ops_about_z_axis():
    """
    D4h 的代表操作（围绕 z 轴），用于 X 点（选择与晶格对齐的主轴）。
    对于简单立方，选择 z 为主轴；若需围绕 x 轴，也可将轴改为 [1,0,0]。
    这里只列点群部分（t=0），Bloch相位=1。
    """
    return [
        ("E",         op_from_cart("E",         np.eye(3))),
        ("C4z",       op_from_cart("C4z",       rotation_matrix_cart([0,0,1], np.pi/2))),
        ("C2z",       op_from_cart("C2z",       rotation_matrix_cart([0,0,1], np.pi))),
        ("C4z^3",     op_from_cart("C4z^3",     rotation_matrix_cart([0,0,1], 3*np.pi/2))),
        ("C2x",       op_from_cart("C2x",       rotation_matrix_cart([1,0,0], np.pi))),   # C2'之一
        ("C2y",       op_from_cart("C2y",       rotation_matrix_cart([0,1,0], np.pi))),   # C2'之一
        ("inversion", op_from_cart("inversion", -np.eye(3))),
        ("S4z",       op_from_cart("S4z",       rotation_matrix_cart([0,0,1], np.pi/2) @ (-np.eye(3)))),
        ("S4z^3",     op_from_cart("S4z^3",     rotation_matrix_cart([0,0,1], 3*np.pi/2) @ (-np.eye(3)))),
        ("sigma_h",   op_from_cart("sigma_h",   np.diag([1,1,-1]))),                      # z→-z
        ("sigma_d1",  op_from_cart("sigma_d1",  np.eye(3) - 2*np.outer([1,1,0],[1,1,0])/2.0)),
        ("sigma_d2",  op_from_cart("sigma_d2",  np.eye(3) - 2*np.outer([1,-1,0],[1,-1,0])/2.0)),
    ]

# ---------- 3) Γ点 Oh 共轭类代表操作（t=0） ----------
ops = [
    ("E",            op_from_cart("E",            np.eye(3))),
    ("C3_[111]",     op_from_cart("C3_[111]",     rotation_matrix_cart([1,1,1], 2*np.pi/3))),
    ("C4z",          op_from_cart("C4z",          rotation_matrix_cart([0,0,1], np.pi/2))),
    ("C2z",          op_from_cart("C2z",          rotation_matrix_cart([0,0,1], np.pi))),
    ("C2_[110]",     op_from_cart("C2_[110]",     rotation_matrix_cart([1,1,0], np.pi))),
    ("inversion",    op_from_cart("inversion",    -np.eye(3))),
    ("S4z",          op_from_cart("S4z",          rotation_matrix_cart([0,0,1], np.pi/2) @ (-np.eye(3)))),
    ("sigma_h",      op_from_cart("sigma_h",      np.diag([1,1,-1]))),  # 镜面 z→-z
    ("sigma_d(110)", op_from_cart("sigma_d(110)", np.eye(3) - 2*np.outer([1,1,0],[1,1,0]) / 2.0)),
]

# ---------- 4) 工具：坐标映射/插值/旋转/内积/正交 ----------
def frac_to_index_centered(coords_frac, grid):
    # 分数坐标在 [-0.5, 0.5)；映射到索引空间 [0, Nx)
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    idx = np.empty_like(coords_frac)
    idx[..., 0] = ((coords_frac[..., 0] + 0.5) % 1.0) * Nx
    idx[..., 1] = ((coords_frac[..., 1] + 0.5) % 1.0) * Ny
    idx[..., 2] = ((coords_frac[..., 2] + 0.5) % 1.0) * Nz
    return idx

def apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart):
    # 分数→笛卡尔（中心原点），应用 R^{-1}(r - t)，再笛卡尔→分数，并周期化回 [-0.5, 0.5)
    x, y, z = grid["x"], grid["y"], grid["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r_frac = np.stack([X, Y, Z], axis=-1)             # (Nx,Ny,Nz,3), centered
    r_cart = r_frac @ A.T
    Rinvc = np.linalg.inv(R_cart)
    r_cart_prime = (r_cart - t_cart) @ Rinvc.T
    r_frac_prime = r_cart_prime @ np.linalg.inv(A).T
    # 周期化到 [-0.5, 0.5)
    r_frac_prime = ((r_frac_prime + 0.5) % 1.0) - 0.5
    return r_frac_prime

def interpolate_field(H, rprime_idx):
    out = np.empty_like(H)
    coords = rprime_idx.transpose(3,0,1,2)  # -> (3,Nx,Ny,Nz)
    for c in range(3):
        out[c] = map_coordinates(H[c], coords, order=3, mode="wrap")
    return out

def rotate_vector_field_axial(H, R_cart):
    # 轴向矢量：w' = det(R) · R · w
    shp = H.shape
    V = H.reshape(3, -1)
    #detR = np.linalg.det(R_cart)
    #Vp = (detR * (R_cart @ V))
    Vp = R_cart @ V
    return Vp.reshape(shp)

def normalize_field(H):
    nrm = np.sqrt(np.sum(np.conj(H) * H).real + 1e-30)
    return H / nrm

def orthonormalize_cluster(cluster):
    X = np.stack([f.reshape(-1) for f in cluster], axis=1)  # (N, m)
    Q, _ = np.linalg.qr(X)
    return [Q[:, i].reshape(cluster[0].shape) for i in range(Q.shape[1])]

def inner_product(H1, H2):
    return np.sum(np.conj(H1) * H2)

def project_onto_cluster(basis, H):
    coeffs = [inner_product(u, H) for u in basis]
    return sum(c * u for c, u in zip(coeffs, basis))

# ---------- 5) 计算字符 ----------
# 取简并簇三条带
cluster = [normalize_field(H_stack[i]) for i in range(n_bands)]
basis = orthonormalize_cluster(cluster)

def rotate_vector_field_with_parity(H, R_cart):
    """
    分量旋转 + 不正操作整体符号修正：
    v' = parity_sign * (R_cart @ v), 其中 parity_sign = sign(det(R_cart))
    对正操作（det=+1）不变；对不正操作（det=-1）整体乘以 -1。
    """
    shp = H.shape
    V = H.reshape(3, -1)
    parity_sign = 1.0 if np.linalg.det(R_cart) > 0 else -1.0
    Vp = parity_sign * (R_cart @ V)
    return Vp.reshape(shp)

def character_for_op(op):
    R_cart, t_frac = op["R_cart"], op["t_frac"]
    # Γ 点：t=0
    t_cart = (np.asarray(t_frac) @ A.T)
    rprime = apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart)
    rprime_idx = frac_to_index_centered(rprime, grid)

    m = len(basis)
    M = np.zeros((m, m), dtype=np.complex128)
    for j, uj in enumerate(basis):
        ujp = interpolate_field(uj, rprime_idx)            # u_j(R^{-1}(r-t))
        ujp = rotate_vector_field_with_parity(ujp, R_cart) # 分量旋转 + 不正操作符号
        # Γ 点：Bloch 相位为 1
        ujp = project_onto_cluster(basis, ujp)
        for i, ui in enumerate(basis):
            M[i, j] = inner_product(ui, ujp)
    return np.trace(M)

def apply_op_to_field_with_k(uj, R_cart, t_frac, k_cart):
    """
    对单个态 uj(r) 应用群操作 Γ(g)：
    1) 坐标：r' = R^{-1}(r - t)（笛卡尔中心坐标）
    2) 插值：uj(R^{-1}(r - t))
    3) 分量：轴向矢量旋转 v' = det(R)·R·v
    4) Bloch 相位：乘 e^{-i k·t_cart}
    """
    t_cart = (np.asarray(t_frac) @ A.T)
    rprime = apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart)
    rprime_idx = frac_to_index_centered(rprime, grid)
    ujp = interpolate_field(uj, rprime_idx)
    # 轴向矢量分量旋转（可选择 parity 或 det(R)*R；X点用与 Γ 相同的轴向规则）
    ujp = rotate_vector_field_axial(ujp, R_cart)
    phase = np.exp(-1j * (k_cart @ t_cart))
    return ujp * phase

def character_for_op_with_k(op, basis, k_cart):
    R_cart, t_frac = op["R_cart"], op["t_frac"]
    m = len(basis)
    M = np.zeros((m, m), dtype=np.complex128)
    for j, uj in enumerate(basis):
        vj = apply_op_to_field_with_k(uj, R_cart, t_frac, k_cart)
        vj_proj = project_onto_cluster(basis, vj)
        for i, ui in enumerate(basis):
            M[i, j] = inner_product(ui, vj_proj)
    return np.trace(M)

def compute_characters_for_clusters_X(H_stack, clusters):
    """
    clusters: 列表，每项为带索引列表，例如 [[0,1],[2,3],[4]]
    输出每个簇在 D4h 代表操作上的字符。
    """
    opsX = d4h_ops_about_z_axis()
    results = []
    for bands in clusters:
        cluster_fields = [normalize_field(H_stack[b]) for b in bands]
        basis = orthonormalize_cluster(cluster_fields)
        chars = []
        for cname, op in opsX:
            chi = character_for_op_with_k(op, basis, k_cart_X)
            chars.append((cname, chi))
        results.append((bands, chars))
    return results

# ---------- 6) 执行并打印 ----------
#print(f"簇大小: {n_bands}, 网格: {Nx}x{Ny}x{Nz} (Γ点, Oh)")
#results = {}
#for cname, op in ops:
#    chi = character_for_op(op)
#    results[cname] = chi
#    print(f"{cname:12s}: chi = {chi.real:.6f} + {chi.imag:.2e}j (≈ {np.round(chi.real)})")

clusters_X = [[0]]  # 对应 1-2 简并、3-4 简并、5 单带（索引从 0 开始）
print(f"\nX 点 (D4h) 特征标: k = {k_frac_X}")
x_results = compute_characters_for_clusters_X(H_stack, clusters_X)
for bands, chars in x_results:
    label = ",".join(str(b+1) for b in bands)
    print(f"簇 bands [{label}] -> 维度 {len(bands)}")
    for cname, chi in chars:
        print(f"  {cname:10s}: chi = {chi.real:.6f} + {chi.imag:.1e}j (≈ {np.round(chi.real)})")

