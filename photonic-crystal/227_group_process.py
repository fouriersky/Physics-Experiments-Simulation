import numpy as np
from scipy.ndimage import map_coordinates

# ---------- 0) FCC 原胞基与中心原点网格 ----------
# irrep227.py: basis_size = sqrt(0.5)，基向量 a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)
# 将 A 设为列矩阵 [a1 a2 a3]，在笛卡尔框架下使用
A = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
], dtype=float) * (1.0 / np.sqrt(2.0))  # 使 |ai|=1/√2，与 basis_size 一致

Ainv = np.linalg.inv(A)

def make_centered_grid(Nx, Ny, Nz):
    x = (np.arange(Nx) / Nx) - 0.5
    y = (np.arange(Ny) / Ny) - 0.5
    z = (np.arange(Nz) / Nz) - 0.5
    return {"Nx": Nx, "Ny": Ny, "Nz": Nz, "x": x, "y": y, "z": z}

# ---------- 1) 数据读取与形状兼容 ----------
def load_npz_hstack(npz_path):
    data = np.load(npz_path)
    H_stack = data["H"]
    # 期望 (n_bands,3,Nx,Ny,Nz)
    if H_stack.ndim == 5 and H_stack.shape[1] == 3:
        pass
    elif H_stack.ndim == 4 and H_stack.shape[0] == 3:
        H_stack = H_stack[None, ...]
    elif H_stack.ndim == 5 and H_stack.shape[0] == 3:
        H_stack = np.moveaxis(H_stack, 1, 0)
    else:
        raise ValueError(f"Unsupported H shape: {H_stack.shape}")
    return H_stack

# ---------- 2) 旋转矩阵与操作构造（以笛卡尔坐标） ----------
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
    # 记录分数版仅用于日志；实际变换使用笛卡尔
    R_frac = Ainv @ R_cart @ A
    R_frac_rounded = np.round(R_frac)
    if np.linalg.norm(R_frac - R_frac_rounded) < 1e-12:
        R_frac = R_frac_rounded
    t_frac = Ainv @ np.asarray(t_cart, dtype=float)
    return {"name": name, "R_frac": R_frac, "R_cart": R_cart, "t_frac": t_frac}

# ---------- 3) 坐标/插值 ----------
def frac_to_index_centered(coords_frac, grid):
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    idx = np.empty_like(coords_frac)
    idx[..., 0] = ((coords_frac[..., 0] + 0.5) % 1.0) * Nx
    idx[..., 1] = ((coords_frac[..., 1] + 0.5) % 1.0) * Ny
    idx[..., 2] = ((coords_frac[..., 2] + 0.5) % 1.0) * Nz
    return idx

def apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart):
    x, y, z = grid["x"], grid["y"], grid["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r_frac = np.stack([X, Y, Z], axis=-1)     # (Nx,Ny,Nz,3) centered frac
    r_cart = r_frac @ A.T                     # frac->cart
    Rinvc = np.linalg.inv(R_cart)
    r_cart_prime = (r_cart - t_cart) @ Rinvc.T
    r_frac_prime = r_cart_prime @ Ainv.T      # cart->frac
    r_frac_prime = ((r_frac_prime + 0.5) % 1.0) - 0.5
    return r_frac_prime

def interpolate_field(H, rprime_idx, order=3):
    out = np.empty_like(H)
    coords = rprime_idx.transpose(3, 0, 1, 2)
    for c in range(3):
        out[c] = map_coordinates(H[c], coords, order=order, mode="wrap")
    return out

# ---------- 4) 分量变换 ----------
def rotate_vector_field_axial(H, R_cart):
    # 轴向矢量：w' = det(R) · R · w
    shp = H.shape
    V = H.reshape(3, -1)
    detR = np.linalg.det(R_cart)
    Vp =  (R_cart @ V)
    return Vp.reshape(shp)

# 默认磁场：轴向矢量
VECTOR_ROTATOR = rotate_vector_field_axial

# ---------- 5) 归一化、正交与投影 ----------
def normalize_field(H):
    nrm = np.sqrt(np.sum(np.conj(H) * H).real + 1e-30)
    return H / nrm

def orthonormalize_cluster(cluster):
    X = np.stack([f.reshape(-1) for f in cluster], axis=1)
    Q, _ = np.linalg.qr(X)
    return [Q[:, i].reshape(cluster[0].shape) for i in range(Q.shape[1])]

def inner_product(H1, H2):
    return np.sum(np.conj(H1) * H2)

def project_onto_cluster(basis, H):
    coeffs = [inner_product(u, H) for u in basis]
    return sum(c * u for c, u in zip(coeffs, basis))

MOTIF_FRAC = [
    np.array([ 0.125,  0.125,  0.125], dtype=float),
    np.array([-0.125, -0.125, -0.125], dtype=float),
]

def wrap_frac(frac):
    # wrap 到 [-0.5, 0.5)
    return ((frac + 0.5) % 1.0) - 0.5

def auto_register_t_frac(R_cart, motif_frac=MOTIF_FRAC):
    """
    在分数坐标中为操作 R 选择 t_frac，使 R_frac·p + t ≡ q (mod 1)，其中 p,q ∈ motif_frac。
    """
    R_frac = Ainv @ R_cart @ A
    best_t = np.zeros(3)
    best_err = 1e9
    for p in motif_frac:
        Rp = R_frac @ p
        Rp_wrapped = wrap_frac(Rp)
        for q in motif_frac:
            t = wrap_frac(q - Rp_wrapped)
            err = np.linalg.norm(wrap_frac(Rp + t) - q)
            if err < best_err:
                best_err = err
                best_t = t
    return best_t  # 分数坐标下的 t

def op_from_cart(name, R_cart, t_frac=(0.0, 0.0, 0.0)):
    # 记录分数版仅用于日志；实际变换使用笛卡尔矩阵、分数平移
    R_frac = Ainv @ R_cart @ A
    R_frac_rounded = np.round(R_frac)
    if np.linalg.norm(R_frac - R_frac_rounded) < 1e-12:
        R_frac = R_frac_rounded
    return {"name": name, "R_frac": R_frac, "R_cart": R_cart, "t_frac": np.asarray(t_frac, dtype=float)}


# ---------- 6) 小群代表操作集合（227，FCC 原胞） ----------
# Γ 点（Oh）
def ops_Oh_Gamma():
    ops = []
    # 列出需要的点群矩阵
    entries = [
        ("E",            np.eye(3)),
        ("C3_[111]",     rotation_matrix_cart([1,1,1], 2*np.pi/3)),
        ("C4z",          rotation_matrix_cart([0,0,1], np.pi/2)),
        ("C2z",          rotation_matrix_cart([0,0,1], np.pi)),
        ("C2_[110]",     rotation_matrix_cart([1,1,0], np.pi)),
        ("inversion",    -np.eye(3)),
        ("S4z",          rotation_matrix_cart([0,0,1], np.pi/2) @ (-np.eye(3))),
        ("sigma_h",      np.diag([1,1,-1])),
        ("sigma_d(110)", np.eye(3) - 2*np.outer([1,1,0],[1,1,0]) / 2.0),
    ]
    for name, R in entries:
        t_frac = auto_register_t_frac(R)  # 关键：按基元自动选择 t
        ops.append((name, op_from_cart(name, R, t_frac=t_frac)))
    return ops

# X 点（D4h），X=(0,1/2,1/2)（FCC 的面心）
def ops_D4h_X():
    # 对边界点一般还需要考虑滑移/螺旋的分数 t；此处先给出点群部分及配准 t_cart
    return [
        ("E",         op_from_cart("E",         np.eye(3),                                 t_cart=(0.0, 0.0, 0.0))),
        ("C4z",       op_from_cart("C4z",       rotation_matrix_cart([0,0,1], np.pi/2),    t_cart=(0.5, 0.5, 0.0))),
        ("C2z",       op_from_cart("C2z",       rotation_matrix_cart([0,0,1], np.pi),      t_cart=(0.0, 0.0, 0.0))),
        ("C4z^3",     op_from_cart("C4z^3",     rotation_matrix_cart([0,0,1], 3*np.pi/2),  t_cart=(0.5, 0.5, 0.0))),
        ("C2x",       op_from_cart("C2x",       rotation_matrix_cart([1,0,0], np.pi),      t_cart=(0.0, 0.0, 0.0))),
        ("C2y",       op_from_cart("C2y",       rotation_matrix_cart([0,1,0], np.pi),      t_cart=(0.0, 0.0, 0.0))),
        ("inversion", op_from_cart("inversion", -np.eye(3),                                 t_cart=(0.0, 0.0, 0.0))),
        ("S4z",       op_from_cart("S4z",       rotation_matrix_cart([0,0,1], np.pi/2) @ (-np.eye(3)), t_cart=(0.5, 0.5, 0.0))),
        ("S4z^3",     op_from_cart("S4z^3",     rotation_matrix_cart([0,0,1], 3*np.pi/2) @ (-np.eye(3)), t_cart=(0.5, 0.5, 0.0))),
        ("sigma_h",   op_from_cart("sigma_h",   np.diag([1,1,-1]),                          t_cart=(0.0, 0.0, 0.0))),
        ("sigma_d1",  op_from_cart("sigma_d1",  np.eye(3) - 2*np.outer([1,1,0],[1,1,0])/2.0, t_cart=(0.5, 0.5, 0.0))),
        ("sigma_d2",  op_from_cart("sigma_d2",  np.eye(3) - 2*np.outer([1,-1,0],[1,-1,0])/2.0, t_cart=(0.5, -0.5, 0.0))),
    ]

# W 点（C2v），W=(1/4,3/4,1/2)
def ops_C2v_W():
    return [
        ("E",        op_from_cart("E",        np.eye(3),                               t_cart=(0.0, 0.0, 0.0))),
        ("C2z",      op_from_cart("C2z",      rotation_matrix_cart([0,0,1], np.pi),    t_cart=(0.0, 0.0, 0.0))),
        ("sigma_v",  op_from_cart("sigma_v",  np.diag([1,-1,1]),                       t_cart=(0.0, 0.0, 0.0))),
        ("sigma_v'", op_from_cart("sigma_v'", np.diag([-1,1,1]),                       t_cart=(0.0, 0.0, 0.0))),
    ]

# L 点（D3d），L=(1/2,1/2,1/2)
def ops_D3d_L():
    R_C3 = rotation_matrix_cart([1,1,1], 2*np.pi/3)
    R_C3_2 = rotation_matrix_cart([1,1,1], 4*np.pi/3)
    return [
        ("E",         op_from_cart("E",          np.eye(3),           t_cart=(0.0, 0.0, 0.0))),
        ("C3_[111]",  op_from_cart("C3_[111]",   R_C3,                t_cart=(0.0, 0.0, 0.0))),
        ("C3^2_[111]",op_from_cart("C3^2_[111]", R_C3_2,              t_cart=(0.0, 0.0, 0.0))),
        ("C2_⊥",      op_from_cart("C2_⊥",       rotation_matrix_cart([1,-1,0], np.pi), t_cart=(0.0, 0.0, 0.0))),
        ("inversion", op_from_cart("inversion",  -np.eye(3),          t_cart=(0.0, 0.0, 0.0))),
        ("S6",        op_from_cart("S6",         R_C3 @ (-np.eye(3)), t_cart=(0.0, 0.0, 0.0))),
    ]


# ---------- 7) 单操作字符计算（带/不带 k） ----------
def apply_op_to_field(uj, grid, R_cart, t_frac):
    t_cart = (np.asarray(t_frac) @ A.T)
    rprime = apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart)
    rprime_idx = frac_to_index_centered(rprime, grid)
    ujp = interpolate_field(uj, rprime_idx, order=3)
    ujp = VECTOR_ROTATOR(ujp, R_cart)
    return ujp  # Γ 点：Bloch 相位=1

def apply_op_to_field_with_k(uj, grid, R_cart, t_frac, k_frac):
    t_cart = (np.asarray(t_frac) @ A.T)
    rprime = apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart)
    rprime_idx = frac_to_index_centered(rprime, grid)
    ujp = interpolate_field(uj, rprime_idx, order=3)
    ujp = VECTOR_ROTATOR(ujp, R_cart)
    # Bloch 相位：e^{-i k·t_cart}; 若 t=0（点群）则相位=1
    k_cart = (np.asarray(k_frac) @ A.T)
    phase = np.exp(-1j * (k_cart @ t_cart))
    return ujp * phase

def character_for_op_in_cluster(op, basis, grid):
    R_cart, t_frac = op["R_cart"], op["t_frac"]
    m = len(basis)
    M = np.zeros((m, m), dtype=np.complex128)
    for j, uj in enumerate(basis):
        vj = apply_op_to_field(uj, grid, R_cart, t_frac)
        vj_proj = project_onto_cluster(basis, vj)
        for i, ui in enumerate(basis):
            M[i, j] = inner_product(ui, vj_proj)
    return np.trace(M)

def character_for_op_in_cluster_with_k(op, basis, grid, k_frac):
    R_cart, t_frac = op["R_cart"], op["t_frac"]
    m = len(basis)
    M = np.zeros((m, m), dtype=np.complex128)
    for j, uj in enumerate(basis):
        vj = apply_op_to_field_with_k(uj, grid, R_cart, t_frac, k_frac)
        vj_proj = project_onto_cluster(basis, vj)
        for i, ui in enumerate(basis):
            M[i, j] = inner_product(ui, vj_proj)
    return np.trace(M)

# ---------- 8) 统一计算接口 ----------
def compute_characters(H_stack, clusters, ops, k_frac=None):
    n_bands, _, Nx, Ny, Nz = H_stack.shape
    grid = make_centered_grid(Nx, Ny, Nz)
    results = []
    for bands in clusters:
        cluster_fields = [normalize_field(H_stack[b]) for b in bands]
        basis = orthonormalize_cluster(cluster_fields)
        chars = []
        for cname, op in ops:
            if k_frac is None:
                chi = character_for_op_in_cluster(op, basis, grid)
            else:
                chi = character_for_op_in_cluster_with_k(op, basis, grid, k_frac)
            chars.append((cname, chi))
        results.append((bands, chars))
    return results

def print_characters(results, title=""):
    if title:
        print(title)
    for bands, chars in results:
        label = ",".join(str(b+1) for b in bands)
        print(f" 簇 bands [{label}] -> 维度 {len(bands)}")
        for cname, chi in chars:
            print(f"  {cname:14s}: chi = {chi.real:.6f} + {chi.imag:.2e}j (≈ {np.round(chi.real)})")

# ---------- 9) 高对称点接口 ----------
def input_227(high_symmetry_point):
    """
    返回 (ops, k_frac)，k_frac 为分数坐标（相对 A 基）。
    支持: 'Gamma'(Oh), 'X'(D4h), 'W'(C2v), 'L'(D3d)
    """
    hs = high_symmetry_point.lower()
    if hs == "gamma":
        return ops_Oh_Gamma(), None
    if hs == "x":
        # FCC: X = (0, 1/2, 1/2)
        return ops_D4h_X(), np.array([0.0, 0.5, 0.5], dtype=float)
    if hs == "w":
        # FCC: W = (1/4, 3/4, 1/2)
        return ops_C2v_W(), np.array([0.25, 0.75, 0.5], dtype=float)
    if hs == "l":
        # FCC: L = (1/2, 1/2, 1/2)
        return ops_D3d_L(), np.array([0.5, 0.5, 0.5], dtype=float)
    raise ValueError(f"Unsupported high-symmetry point: {high_symmetry_point}")

# ---------- 10) 运行示例（按需修改路径与簇） ----------
if __name__ == "__main__":
    # 示例：X 点 D4h，分析若干一维簇
    npz_path = "./data_227/G_3-8.npz"  # 请替换为你的文件
    bands_clusters = [[0,1,2]]           # 例：三条单带
    high_symmetry_point = "gamma"

    H_stack = load_npz_hstack(npz_path)
    ops, k_frac = input_227(high_symmetry_point)

    if k_frac is None:
        results = compute_characters(H_stack, bands_clusters, ops, k_frac=None)
    else:
        results = compute_characters(H_stack, bands_clusters, ops, k_frac=k_frac)
    print_characters(results, title=f"{high_symmetry_point} point character (Fd-3m)")