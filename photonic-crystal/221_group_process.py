import numpy as np
from scipy.ndimage import map_coordinates

# ---------- 0) 基与坐标网格（simple cubic, #225, 用中心原点） ----------
A = np.eye(3)  # simple cubic, a=1

def make_centered_grid(Nx, Ny, Nz):
    x = (np.arange(Nx) / Nx) - 0.5
    y = (np.arange(Ny) / Ny) - 0.5
    z = (np.arange(Nz) / Nz) - 0.5
    return {"Nx": Nx, "Ny": Ny, "Nz": Nz, "x": x, "y": y, "z": z}

# ---------- 1) 数据读取与形状兼容 ----------
def load_npz_hstack(npz_path):
    data = np.load(npz_path)
    H_stack = data["H"]
    # 兼容常见维度：期望 (n_bands,3,Nx,Ny,Nz)
    if H_stack.ndim == 5 and H_stack.shape[1] == 3:
        pass
    elif H_stack.ndim == 4 and H_stack.shape[0] == 3:
        H_stack = H_stack[None, ...]
    elif H_stack.ndim == 5 and H_stack.shape[0] == 3:
        H_stack = np.moveaxis(H_stack, 1, 0)
    else:
        raise ValueError(f"Unsupported H shape: {H_stack.shape}")
    return H_stack

# ---------- 2) 旋转矩阵与操作构造 ----------
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
    R_frac = Ainv @ R_cart @ A
    R_frac_rounded = np.round(R_frac)
    if np.linalg.norm(R_frac - R_frac_rounded) < 1e-12:
        R_frac = R_frac_rounded
    t_frac = Ainv @ np.asarray(t_cart, dtype=float)
    return {"name": name, "R_frac": R_frac, "R_cart": R_cart, "t_frac": t_frac}

# ---------- 3) 中心原点坐标变换/插值 ----------
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
    r_frac = np.stack([X, Y, Z], axis=-1)             # (Nx,Ny,Nz,3), centered
    r_cart = r_frac @ A.T
    Rinvc = np.linalg.inv(R_cart)
    r_cart_prime = (r_cart - t_cart) @ Rinvc.T
    r_frac_prime = r_cart_prime @ np.linalg.inv(A).T
    r_frac_prime = ((r_frac_prime + 0.5) % 1.0) - 0.5
    return r_frac_prime

def interpolate_field(H, rprime_idx, order=3):
    out = np.empty_like(H)
    coords = rprime_idx.transpose(3, 0, 1, 2)  # (3,Nx,Ny,Nz)
    for c in range(3):
        out[c] = map_coordinates(H[c], coords, order=order, mode="wrap")
    return out

# ---------- 4) 分量变换：轴向矢量或带奇偶校正 ----------
def rotate_vector_field_axial(H, R_cart):
    # 轴向矢量：w' = det(R) · R · w, but here we treat H field as polar vector to be consistent with character table
    shp = H.shape
    V = H.reshape(3, -1)
    detR = np.linalg.det(R_cart)
    Vp = (detR * (R_cart @ V))
    #Vp = R_cart @ V
    return Vp.reshape(shp)

def rotate_vector_field_with_parity(H, R_cart):
    # 极/统一约定：v' = sign(det(R)) · (R v)
    shp = H.shape
    V = H.reshape(3, -1)
    parity_sign = 1.0 
    Vp = parity_sign * (R_cart @ V)
    return Vp.reshape(shp)

# 选择分量变换器（Gamma 点 T1g/T1u 的奇偶可以通过这里切换）
#VECTOR_ROTATOR = rotate_vector_field_axial  
VECTOR_ROTATOR = rotate_vector_field_with_parity

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

# ---------- 6) 小群代表操作集合 ----------
def ops_Oh_Gamma():
    return [
        ("E",            op_from_cart("E",            np.eye(3))),
        ("C3_[111]",     op_from_cart("C3_[111]",     rotation_matrix_cart([1,1,1], 2*np.pi/3))),
        ("C4z",          op_from_cart("C4z",          rotation_matrix_cart([0,0,1], np.pi/2))),
        ("C2z",          op_from_cart("C2z",          rotation_matrix_cart([0,0,1], np.pi))),
        ("C2_[110]",     op_from_cart("C2_[110]",     rotation_matrix_cart([1,1,0], np.pi))),
        ("inversion",    op_from_cart("inversion",    -np.eye(3))),
        ("S4z",          op_from_cart("S4z",          rotation_matrix_cart([0,0,1], np.pi/2) @ (-np.eye(3)))),
        ("sigma_h",      op_from_cart("sigma_h",      np.diag([1,1,-1]))),
        ("sigma_d(110)", op_from_cart("sigma_d(110)", np.eye(3) - 2*np.outer([1,1,0],[1,1,0]) / 2.0)),
        ("IC3_[111]",     op_from_cart("C3_[111]",     rotation_matrix_cart([1,1,1], 2*np.pi/3)@ (-np.eye(3))))
    ]

def ops_D4h_X():
    return [
        ("E",         op_from_cart("E",         np.eye(3))),
        ("C4z",       op_from_cart("C4z",       rotation_matrix_cart([0,0,1], np.pi/2))),
        ("C2z",       op_from_cart("C2z",       rotation_matrix_cart([0,0,1], np.pi))),
        ("C4z^3",     op_from_cart("C4z^3",     rotation_matrix_cart([0,0,1], 3*np.pi/2))),
        ("C2x",       op_from_cart("C2x",       rotation_matrix_cart([1,0,0], np.pi))),
        ("C2y",       op_from_cart("C2y",       rotation_matrix_cart([0,1,0], np.pi))),
        ("inversion", op_from_cart("inversion", -np.eye(3))),
        ("S4z",       op_from_cart("S4z",       rotation_matrix_cart([0,0,1], np.pi/2) @ (-np.eye(3)))),
        ("S4z^3",     op_from_cart("S4z^3",     rotation_matrix_cart([0,0,1], 3*np.pi/2) @ (-np.eye(3)))),
        ("sigma_h",   op_from_cart("sigma_h",   np.diag([1,1,-1]))),
        ("sigma_d1",  op_from_cart("sigma_d1",  np.eye(3) - 2*np.outer([1,1,0],[1,1,0])/2.0)),
        ("sigma_d2",  op_from_cart("sigma_d2",  np.eye(3) - 2*np.outer([1,-1,0],[1,-1,0])/2.0)),
    ]

def ops_C4v_Delta():
    return [
        ("E",        op_from_cart("E",        np.eye(3))),
        ("C4z",      op_from_cart("C4z",      rotation_matrix_cart([0,0,1], np.pi/2))),
        ("C2z",      op_from_cart("C2z",      rotation_matrix_cart([0,0,1], np.pi))),
        ("C4z^3",    op_from_cart("C4z^3",    rotation_matrix_cart([0,0,1], 3*np.pi/2))),
        ("sigma_v",  op_from_cart("sigma_v",  np.diag([1,-1,1]))),  # xz 镜面
        ("sigma_v'", op_from_cart("sigma_v'", np.diag([-1,1,1]))),  # yz 镜面
    ]

def ops_C2v_Sigma():
    axis = np.array([1.0, 1.0, 0.0])
    R_C2 = rotation_matrix_cart(axis, np.pi)
    n1 = np.array([1.0, -1.0, 0.0]); n1 /= np.linalg.norm(n1)
    n2 = np.array([0.0, 0.0, 1.0])
    R_sigma1 = np.eye(3) - 2.0 * np.outer(n1, n1)  # x=y plane
    R_sigma2 = np.diag([1.0, 1.0, -1.0])           # z=0 plane

    return [
        ("E",        op_from_cart("E",        np.eye(3))),
        ("C2_[110]", op_from_cart("C2_[110]", R_C2)),
        ("sigma_v(x=y)",  op_from_cart("sigma_v(x=y)",  R_sigma1)),
        ("sigma_v(z=0)",  op_from_cart("sigma_v(z=0)",  R_sigma2)),
    ]

def ops_Oh_R():
     return ops_Oh_Gamma()

def ops_C2v_S():
    return ops_C2v_Sigma()

# ---------- 7) 单操作字符计算（带/不带 k） ----------
def apply_op_to_field(uj, grid, R_cart, t_frac):
    t_cart = (np.asarray(t_frac) @ A.T)
    rprime = apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart)
    rprime_idx = frac_to_index_centered(rprime, grid)
    ujp = interpolate_field(uj, rprime_idx, order=3)
    ujp = VECTOR_ROTATOR(ujp, R_cart)
    return ujp  # Γ点：Bloch 相位=1

def apply_op_to_field_with_k(uj, grid, R_cart, t_frac, k_cart):
    t_cart = (np.asarray(t_frac) @ A.T)
    rprime = apply_symmetry_to_coords_cart_centered(grid, R_cart, t_cart)
    rprime_idx = frac_to_index_centered(rprime, grid)
    ujp = interpolate_field(uj, rprime_idx, order=3)
    ujp = VECTOR_ROTATOR(ujp, R_cart)
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

def character_for_op_in_cluster_with_k(op, basis, grid, k_cart):
    R_cart, t_frac = op["R_cart"], op["t_frac"]
    m = len(basis)
    M = np.zeros((m, m), dtype=np.complex128)
    for j, uj in enumerate(basis):
        vj = apply_op_to_field_with_k(uj, grid, R_cart, t_frac, k_cart)
        vj_proj = project_onto_cluster(basis, vj)
        for i, ui in enumerate(basis):
            M[i, j] = inner_product(ui, vj_proj)
    return np.trace(M)

# ---------- 8) 统一计算接口 ----------
def compute_characters(H_stack, clusters, ops, k_frac=None):
    n_bands, _, Nx, Ny, Nz = H_stack.shape
    grid = make_centered_grid(Nx, Ny, Nz)
    results = []
    k_cart = None
    if k_frac is not None:
        k_cart = (np.asarray(k_frac) @ A.T)
    for bands in clusters:
        cluster_fields = [normalize_field(H_stack[b]) for b in bands]
        basis = orthonormalize_cluster(cluster_fields)
        chars = []
        for cname, op in ops:
            if k_cart is None:
                chi = character_for_op_in_cluster(op, basis, grid)
            else:
                chi = character_for_op_in_cluster_with_k(op, basis, grid, k_cart)
            chars.append((cname, chi))
        results.append((bands, chars))
    return results

def print_characters(results, title=""):
    if title:
        print(title)
    for bands, chars in results:
        label = ",".join(str(b+1) for b in bands)
        print(f" bands [{label}] -> dimension {len(bands)}")
        for cname, chi in chars:
            print(f"  {cname:12s}: chi = {chi.real:.6f} + {chi.imag:.2e}j (≈ {np.round(chi.real)})")

def input(high_symmetry_point):
    """
    根据高对称点名称返回小群操作集合与 k_frac。
    支持: 'Gamma' (Oh), 'X' (D4h), 'Delta' (C4v), 'Sigma' (C2v)
    """
    hs = high_symmetry_point.lower()
    if hs == "gamma":
        return ops_Oh_Gamma(), None
    if hs == "x":
        # 约定 X=(0,0,0.5)；若你的 MPB 选择 X=(0.5,0,0)，请替换为 np.array([0.5,0,0])
        return ops_D4h_X(), np.array([0.0, 0.0, 0.5], dtype=float)
    if hs == "delta":
        # Δ=(0,0,0.25)
        return ops_C4v_Delta(), np.array([0.0, 0.0, 0.25], dtype=float)
    if hs == "sigma":
        # Σ=(0.25,0.25,0.0)
        return ops_C2v_Sigma(), np.array([0.25, 0.25, 0.0], dtype=float)
    if hs == "r":
        return ops_Oh_R(), np.array([0.50, 0.50, 0.50], dtype= float)
    if hs == "s":
        return ops_C2v_S(), np.array([0.20, 0.20, 0.50])
    raise ValueError(f"Unsupported high-symmetry point: {high_symmetry_point}")


if __name__ == "__main__":
    npz_path = "./data_225/225_R_123_456.npz"
    high_symmetry_point = "r"
    #bands_clusters = [[0],[1],[2],[3],[4],[5]]  
    bands_clusters = [[0,1,2],[3,4,5]]
    #bands_clusters = [[0,1,2]]

    H_stack = load_npz_hstack(npz_path)
    ops, k_frac = input(high_symmetry_point)

    if k_frac is None:
        results = compute_characters(H_stack, bands_clusters, ops, k_frac=None)
        print_characters(results, title="Γ point (Oh) character")
    else:
        results = compute_characters(H_stack, bands_clusters, ops, k_frac=k_frac)
        print_characters(results, title=f"{high_symmetry_point} point character: k={k_frac}")