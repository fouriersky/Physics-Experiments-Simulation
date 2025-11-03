"""
this module aims at generating the phonon dispersion relationship
(1) displace atoms,
(2) solving the Force_constant matrix,
(3) do fourier transform to get dynamic matrix 

more interesting is the finite temperature case
phonon dispersion may change at finite temperature compared with 0K 
"""

"""
用 2x2x2 超胞有限位移法从 energy_and_forces(...) 得到声子色散。
依赖: numpy, matplotlib, 以及你现有的 DirectLAMMPSEAMself 类（或其他同接口类）。
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from itertools import product
import time

# --- 物理常数与单位换算 ---
eV_to_J = 1.602176634e-19      # 1 eV in J  1 meV = 0.242 THz = 8.066 cm^−1
angstrom_to_m = 1e-10
A2_to_m2 = angstrom_to_m**2
amu_to_kg = 1.66053906660e-27

# -------------------------
# 1) 超胞 / 原胞 mapping
# -------------------------
def build_supercell_mapping(nbasis, nx, ny, nz):
    """
    返回超胞中每个原子索引对应 (basis_index, cell_shift_index)
    其中 cell_shift_index 是 (ix,iy,iz) 中的三元组。
    假定 build_supercell 的排列顺序是:
       for ix,iy,iz in product(range(nx),...):
           for b in range(nbasis):
               append atom
    """
    mapping = []
    for ix,iy,iz in product(range(nx), range(ny), range(nz)):
        for b in range(nbasis):
            mapping.append((b, (ix,iy,iz)))
    return mapping  # 长度 N_super

def cell_shift_to_vector(shift, prim_cell):
    """shift=(ix,iy,iz) -> cartesian vector R = ix*a1 + iy*a2 + iz*a3"""
    ix,iy,iz = shift
    # prim_cell is 3x3 with columns as lattice vectors or rows? we assume cell is 3x3 matrix with basis vectors as rows in your code,
    # but build_supercell returned diag([nx*a,...]) and positions as cartesian. To be safe, treat prim_cell columns as lattice vectors.
    # We'll compute R = prim_cell @ [ix,iy,iz]
    return prim_cell.dot(np.array([ix,iy,iz], dtype=float))

# -------------------------
# 2) 计算超胞力常数（有限位移）
# -------------------------
def compute_force_constants_supercell(calc, positions_super, cell_super, nbasis, nx,ny,nz, delta=1e-3):
    """
    - calc.energy_and_forces(positions, cell) -> (E, forces)
    - 返回“超胞力常数矩阵” Phi_super: shape (N_super,3,N_super,3), 单位 eV/Å^2
    - 同时返回 mapping: list[(ib, (ix,iy,iz))]，标识每个超胞原子属于原胞中的哪个基元原子及其平移
    - 以及 prim_cell（原胞晶格向量，列为 a1,a2,a3）
    - 同时返回 mapping: list[(ib, (ix,iy,iz))]，标识每个超胞原子属于“构建超胞所用基胞”的哪个基元及其平移
    - 以及：
        conv_cell: 构建超胞所用 conventional cell,列为 a1,a2,a3
        prim_cell_prim: 对应的 fcc primitive cell,列为 (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
    """
    N_super = len(positions_super)
    mapping = build_supercell_mapping(nbasis, nx, ny, nz)  # 长度 N_super

    # 原胞晶格向量（列向量），由超胞 cell 拆分
    conv_cell = cell_super.copy()
    conv_cell[:,0] /= nx
    conv_cell[:,1] /= ny
    conv_cell[:,2] /= nz
    # fcc primitive cell = conv_cell @ T，其中 T 的列为 (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
    T_fcc_to_prim = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    prim_cell_prim = conv_cell @ T_fcc_to_prim  # 列向量为 primitive a1',a2',a3'

    Phi_super = np.zeros((N_super,3,N_super,3), dtype=float)

    print(f"[INFO] Supercell size: {nx}x{ny}x{nz}, N_super = {N_super}")
    t0 = time.time()
    for j in range(N_super):
        for beta in range(3):
            disp = np.zeros_like(positions_super)
            disp[j, beta] = delta
            E_plus, F_plus = calc.energy_and_forces(positions_super + disp, cell_super)
            E_minus, F_minus = calc.energy_and_forces(positions_super - disp, cell_super)
            dF = (F_plus - F_minus) / (2.0 * delta)
            # Phi(iα, jβ) = - ∂F_iα / ∂u_jβ
            for i in range(N_super):
                Phi_super[i, :, j, beta] = -dF[i, :]
        if (j+1) % 5 == 0 or j == N_super-1:
            print(f"  computed displacements for atom {j+1}/{N_super}")
    t1 = time.time()
    print(f"[INFO] Finite displacement done in {t1-t0:.1f} s")

    return Phi_super, mapping, conv_cell, prim_cell_prim

# -------------------------
# 3) 构造动力学矩阵并计算频率
# -------------------------
def build_D_q_from_prim_IFC(Phi_prim, R_list, prim_cell, masses_basis, q_cart):
    """
    Phi_prim: dict (ib,jb,R_shift) -> 3x3 (eV/Å^2)
    R_list: list of R_shift tuples (ix,iy,iz) - integer multiples of prim_cell
    prim_cell: 3x3 (cartesian lattice vectors columns)
    masses_basis: array length nbasis in amu
    q_cart: wavevector in cartesian (1/Å)
    返回: D(q)  (3N x 3N) complex, units: eV/Å^2 / amu_massnormalized (we'll convert later)
    """
    nbasis = len(masses_basis)
    N = nbasis
    D = np.zeros((3*N, 3*N), dtype=complex)
    # precompute cartesian R vectors
    R_cart_list = [cell_shift_to_vector(R, prim_cell) for R in R_list]  # in Å
    # loop keys
    for (ib, jb, R_shift), block in Phi_prim.items():
        # phase e^{i q·R}
        # R_cart in Å, q_cart in 1/Å => q·R dimensionless
        R_cart = cell_shift_to_vector(R_shift, prim_cell)
        phase = np.exp(1j * np.dot(q_cart, R_cart))
        i0 = 3*ib
        j0 = 3*jb
        # add block * phase / sqrt(m_i m_j)
        mfac = 1.0 / np.sqrt(masses_basis[ib] * masses_basis[jb])
        D[i0:i0+3, j0:j0+3] += block * phase * mfac
    return D  # units: eV/Å^2 / sqrt(amu*amu) -> eV/Å^2 / amu

def build_D_q_from_supercell_IFC(Phi_super, mapping, prim_cell, masses_basis, q_cart):
    """
    由“超胞 IFC”直接构造 D(q)（在此处完成对原胞的折叠）:
      - 选参考原胞（shift=(0,0,0)）中的每个基元原子 ib 为“i”
      - 对超胞所有原子 j 累加: e^{i q·R_j} Phi(i,j) / sqrt(m_i m_jb)
    参数:
      Phi_super: (Ns,3,Ns,3) 超胞 IFC
      mapping: 长度 Ns 的列表，每项为 (ib, (ix,iy,iz))
      prim_cell: 3x3，列为 a1,a2,a3 (Å)
      masses_basis: (nbasis,) amu
      q_cart: (3,) 1/Å
    返回:
      D: (3*nbasis, 3*nbasis) 复数矩阵（单位 eV/Å^2 / amu）
    """
    Ns = Phi_super.shape[0]
    nbasis = len(masses_basis)

    # 参考原胞中各基元原子在超胞数组中的索引（按 ib 排序）
    ref_idx_by_ib = {}
    for idx, (ib, shift) in enumerate(mapping):
        if shift == (0,0,0) and (ib not in ref_idx_by_ib):
            ref_idx_by_ib[ib] = idx
    if len(ref_idx_by_ib) != nbasis:
        raise RuntimeError("未能在超胞中定位到完整的参考原胞（shift=(0,0,0)）基元原子集合。")

    # 预计算每个超胞原子 j 的 R_cart
    R_cart_all = []
    jb_all = []
    for j in range(Ns):
        jb, shift_j = mapping[j]
        R_cart_all.append(cell_shift_to_vector(shift_j, prim_cell))
        jb_all.append(jb)
    R_cart_all = np.array(R_cart_all, dtype=float)
    jb_all = np.array(jb_all, dtype=int)
    D = np.zeros((3*nbasis, 3*nbasis), dtype=complex)

    # 循环 ib（行块）与 j over supercell（列方贡献到 jb 列）
    phases = np.exp(1j * (R_cart_all @ q_cart))  # (Ns,)
    for ib in range(nbasis):
        iref = ref_idx_by_ib[ib]
        i0 = 3*ib
        for j in range(Ns):
            jb = jb_all[j]
            j0 = 3*jb
            block = Phi_super[iref, :, j, :]  # 3x3
            mfac = 1.0 / np.sqrt(masses_basis[ib] * masses_basis[jb])
            D[i0:i0+3, j0:j0+3] += block * phases[j] * mfac

    return D


def eigfreqs_from_D(D):
    """
    输入 D (3N x 3N) in units eV/Å^2 / amu  (see build_D_q_from_prim_IFC)
    返回频率 (THz)，按升序排列每 q 的 3N 个频率（非负）
    """
    # convert to SI: multiply D by factor (eV->J)/(Å^2->m^2) / (amu->kg)
    conv = (eV_to_J / A2_to_m2) / amu_to_kg  # (J/m^2) / kg => 1/s^2
    D_SI = D * conv  # complex
    # hermitianize (numerical)
    D_SI = 0.5 * (D_SI + D_SI.conj().T)
    w2, _ = eigh(D_SI)
    # numerical negative set to zero
    w2_real = np.real(w2)
    w2_real[w2_real < 0] = 0.0
    w = np.sqrt(w2_real)  # rad/s
    freqs_THz = w / (2.0 * np.pi * 1e12)
    return np.sort(freqs_THz)

# -------------------------
# 4) 高对称路径（fractional w.r.t primitive reciprocal lattice）
# -------------------------
def reciprocal_vectors(cell_prim):
    """给出 primitive reciprocal basis vectors b1,b2,b3 (cartesian) in 1/Å"""
    # cell_prim columns are a1,a2,a3 (Å)
    a1 = cell_prim[:,0]; a2 = cell_prim[:,1]; a3 = cell_prim[:,2]
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2*np.pi * np.cross(a2, a3) / V
    b2 = 2*np.pi * np.cross(a3, a1) / V
    b3 = 2*np.pi * np.cross(a1, a2) / V
    B = np.vstack([b1, b2, b3]).T  # columns are b1,b2,b3
    return B

def path_G_X_L_G(nseg=40):
    """
    Diamond primitive reciprocal basis 下的连续路径（分数坐标 w.r.t B_prim）:
      Δ: Γ -> X      (X_prim = (1/2, 0, 1/2))
      Σ: K -> Γ      (K_prim = (3/8, 3/8, 3/4))
      Λ: Γ -> L      (L_prim = (1/2, 1/2, 1/2))
    注意：这里按你的要求“连续绘制”，三段直接首尾拼接：Γ→X，接着 K→Γ，接着 Γ→L。
    返回:
      path_frac: (N,3) primitive 分数坐标
      labels: ['Γ','X','Γ','L']  对应 ticks: start, Δ端点, Σ端点(回到Γ), Λ端点
    """
    G  = np.array([0.0,   0.0,   0.0  ])
    Xp = np.array([0.50,  0.0,  0.50 ])   # (0,1/2,1/2)conv -> (1/4,1/4,1/2)prim
    Kp = np.array([0.0, 0.5,  0.5])  # (3/8,3/8,0)conv -> (3/8,3/16,3/16)prim
    Lp = np.array([0.5,   0.5,   0.5  ])

    # 段1：Γ->X（含端点）
    seg1 = np.linspace(G,  Xp, nseg+1, endpoint=True)
    # 段2：K->Γ（含端点，与上一段不连续是预期的）
    seg2 = np.linspace(Kp, G,  nseg+1, endpoint=True)
    # 段3：Γ->L（含端点）
    seg3 = np.linspace(G,  Lp, nseg+1, endpoint=True)

    path = np.vstack([seg1, seg2, seg3])
    labels = [r'$\Gamma$', 'X', r'$\Gamma$', 'L']
    return path, labels

def _inv_cell(M):
    return np.linalg.inv(M)

def _frac_wrt_cell(r_cart, cell):
    # r_cart: (...,3), cell: 3x3 (columns are lattice vectors)
    G = _inv_cell(cell)
    return np.dot(r_cart, G.T)

def _wrap01(x):
    # wrap to [0,1)
    return x - np.floor(x)

def build_D_q_from_supercell_IFC_primitive2(Phi_super, positions_super, prim_cell_prim,
                                            masses_2, q_cart, tol=1e-3):
    """
    直接用 primitive 格矢分解超胞中每个原子 j：
      positions_super[j] = prim_cell_prim @ (t_j + u_j)
      其中 t_j 为整数三元组（primitive 晶格平移），u_j ∈ {A(0,0,0), B(1/4,1/4,1/4)}
    选参考单胞 t=(0,0,0) 中的 A/B 两个原子分别作为 i_ref，构造:
      D_ab(q) = Σ_j  Phi(i_ref(a), j) * exp(i q · (prim_cell_prim @ t_j)) / sqrt(m_a m_b(j))
    参数
      Phi_super: (Ns,3,Ns,3) 超胞 IFC，单位 eV/Å^2
      positions_super: (Ns,3) Å
      prim_cell_prim: 3x3 primitive 晶格（列向量）
      masses_2: (2,) amu，对应 A/B 的质量
      q_cart: (3,) 1/Å
    返回
      D: (6,6) complex
    """
    Ns = Phi_super.shape[0]

    # 1) 以 primitive 晶格分解：t_j 整数平移，u_j 基元内分数坐标
    fprim = _frac_wrt_cell(np.asarray(positions_super, float), prim_cell_prim)  # (Ns,3)
    t_int = np.floor(fprim + 1e-9).astype(int)                                  # (Ns,3)
    u_frac = _wrap01(fprim - t_int)                                             # (Ns,3)

    # 2) 分类到 A/B
    bprim = np.zeros(Ns, dtype=int)       # 0->A, 1->B
    for j in range(Ns):
        b, _ = _closest_basis_primitive(u_frac[j], tol=tol)
        bprim[j] = b

    # 3) 找参考原胞 t=(0,0,0) 中的 A/B i_ref
    iref_A = iref_B = None
    for j in range(Ns):
        if np.all(t_int[j] == 0):
            if bprim[j] == 0 and iref_A is None:
                iref_A = j
            if bprim[j] == 1 and iref_B is None:
                iref_B = j
        if iref_A is not None and iref_B is not None:
            break
    if iref_A is None or iref_B is None:
        raise RuntimeError("未在 primitive 参考单胞(t=0)内找到 A 或 B 原子作为 i_ref，请检查坐标/格矢。")

    # 4) 相位：只用晶格平移 t_j（基元内位移已由 i_ref 与分类吸收）
    R_cart = t_int.astype(float) @ prim_cell_prim.T   # (Ns,3)
    phases = np.exp(1j * (R_cart @ q_cart))          # (Ns,)

    # 5) 组装 D(6x6)
    D = np.zeros((6, 6), dtype=complex)
    for a, iref in enumerate([iref_A, iref_B]):  # a=0(A),1(B)
        i0 = 3 * a
        for j in range(Ns):
            b = bprim[j]
            j0 = 3 * b
            block = Phi_super[iref, :, j, :]  # 3x3
            mfac = 1.0 / np.sqrt(masses_2[a] * masses_2[b])
            D[i0:i0+3, j0:j0+3] += block * phases[j] * mfac

    return D

def build_D_q_from_supercell_IFC_primitive(Phi_super, positions_super, prim_cell_prim,
                                           basis_fracs_prim, masses_basis, q_cart, tol=1e-4):
    """
    通用：从“超胞 IFC + primitive 基矢 + primitive 基元坐标”构造 D(q)
      D_bb'(q) = sum_T Phi(0b, T b') * exp(i q · (T + τ_b' − τ_b)) / sqrt(m_b m_b')
    参数
      Phi_super: (Ns,3,Ns,3) 由有限位移得到的超胞 IFC（在笛卡尔坐标）
      positions_super: (Ns,3) 超胞原子坐标（Å）
      prim_cell_prim: 3x3 primitive 实空间晶格（列向量，Å）
      basis_fracs_prim: (nbasis,3) primitive 单胞内基元分数坐标（如 C: [[0,0,0],[1/4,1/4,1/4]]；Al: [[0,0,0]]）
      masses_basis: (nbasis,) amu
      q_cart: (3,) 1/Å
    返回
      D: (3*nbasis, 3*nbasis) complex
    """
    Ns = Phi_super.shape[0]
    basis_fracs_prim = np.asarray(basis_fracs_prim, float)
    nbasis = basis_fracs_prim.shape[0]

    # 1) 把超胞每个原子分解为 primitive: f = T + u, T ∈ Z^3, u ∈ [0,1)^3
    fprim = _frac_wrt_cell(np.asarray(positions_super, float), prim_cell_prim)  # (Ns,3)
    T_int = np.floor(fprim + 1e-9).astype(int)
    u_frac = _wrap01(fprim - T_int)
    # 2) 分类：把 u_frac 归并到最近的基元坐标 basis_fracs_prim[k]
    b_of_j = np.empty(Ns, dtype=int)
    for j in range(Ns):
        dists = [ _torus_dist(u_frac[j], basis_fracs_prim[k]) for k in range(nbasis) ]
        k = int(np.argmin(dists))
        if dists[k] > tol:
            raise RuntimeError(f"原子 {j} 无法匹配 primitive 基元坐标，最小误差={dists[k]:.3e} > tol={tol}")
        b_of_j[j] = k

    # 3) 在参考 primitive 单胞 T=(0,0,0) 中，为每个基元 b 找一个 i_ref
    iref_by_b = {}
    for j in range(Ns):
        if np.all(T_int[j] == 0):
            b = b_of_j[j]
            if b not in iref_by_b:
                # 要求其基元内分数坐标确属该基元（容差判断）
                if _torus_dist(u_frac[j], basis_fracs_prim[b]) <= tol:
                    iref_by_b[b] = j
        if len(iref_by_b) == nbasis:
            break
    if len(iref_by_b) != nbasis:
        missing = [b for b in range(nbasis) if b not in iref_by_b]
        raise RuntimeError(f"参考 primitive 单胞内缺少基元索引: {missing}")

    # 4) 组装 D(q)，相位含 τ_b' − τ_b
    D = np.zeros((3*nbasis, 3*nbasis), dtype=complex)
    # 预先把 R_tot = T + (τ_b' − τ_b) 映射到笛卡尔
    tau = basis_fracs_prim  # (nbasis,3)
    for b in range(nbasis):               # 行块（参考原子）
        iref = iref_by_b[b]
        i0 = 3*b
        for j in range(Ns):               # 列贡献
            bp = b_of_j[j]
            j0 = 3*bp
            # R_tot_frac = T_int[j] + (tau[bp] - tau[b])
            Rtot_frac = T_int[j].astype(float) + (tau[bp] - tau[b])
            Rtot_cart = Rtot_frac @ prim_cell_prim.T   # Å
            phase = np.exp(1j * (Rtot_cart @ q_cart))
            mfac = 1.0 / np.sqrt(masses_basis[b] * masses_basis[bp])
            D[i0:i0+3, j0:j0+3] += Phi_super[iref, :, j, :] * phase * mfac
    return D

def _closest_basis_primitive(u_frac, tol=1e-3):
    """
    在 primitive 单胞中将分数坐标 u_frac ∈ [0,1)^3 归类到:
      A: (0,0,0)
      B: (1/4,1/4,1/4)
    返回 (b, u_std) 其中 b∈{0,1}, u_std 为匹配到的标准分数坐标
    使用周期距离：min(|Δ|, 1-|Δ|) 分量求和
    """
    A = np.array([0.0, 0.0, 0.0], dtype=float)
    B = np.array([0.25, 0.25, 0.25], dtype=float)

    def _dist(fr, tgt):
        d = np.abs(_wrap01(fr - tgt))
        d = np.minimum(d, 1.0 - d)
        return float(d.sum())

    dA = _dist(u_frac, A)
    dB = _dist(u_frac, B)
    if min(dA, dB) > tol:
        # 容错：返回最近者，但提示
        return (0 if dA <= dB else 1), (A if dA <= dB else B)
    return (0 if dA <= dB else 1), (A if dA <= dB else B)

def _torus_dist(fr, tgt):
    d = np.abs(_wrap01(fr - tgt))
    return np.minimum(d, 1.0 - d).sum()

def main(use_diamond = True,delta = 1e-3,nx = 2,ny = 2,nz = 2,npts_seg = 40,use_primitive_2  = True,filepath=None):        
    from potential import DirectLAMMPSEAMself as LMPCalcAl  
    from potential import DirectLAMMPSLCBOPself as LMPCalccarbon
    from opt_method import make_fcc, make_diamond, build_supercell

    if not use_diamond:
        # Al example (EAM)
        calc = LMPCalcAl(
            eam_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\Al_zhou.eam.alloy",
            lmp_cmd=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
            element="Al",
            pair_style="eam/alloy",
            mass=26.981538,
            keep_tmp_files=False
        )
        lat = make_fcc(4.05)
    else:
        calc = LMPCalccarbon(
            lcbop_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
            lmp_cmd=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
            element="C",
            pair_style="lcbop",
            mass=12.011,
            keep_tmp_files=False
        )
        lat = make_diamond(3.56)

    positions_super, cell_super, symbols_super = build_supercell(lat, nx, ny, nz)
    N_super = len(positions_super)
    nbasis = len(lat.basis_frac)  # atoms per primitive cell
    print(f"Primitive basis atoms: {nbasis}, Supercell atoms: {N_super}")

    if use_diamond:
        basis_fracs_prim = np.array([[0.0, 0.0, 0.0],
                                     [0.25,0.25,0.25]], dtype=float)
        masses_basis_prim = np.array([12.011, 12.011], dtype=float)
    else:
        basis_fracs_prim = np.array([[0.0, 0.0, 0.0]], dtype=float)
        masses_basis_prim = np.array([26.981538], dtype=float)


    Phi_super, mapping, conv_cell, prim_cell_prim = compute_force_constants_supercell(
        calc, positions_super, cell_super, nbasis, nx,ny,nz, delta=delta
    )

    frac_path, _ = path_G_X_L_G(nseg=npts_seg)
    B = reciprocal_vectors(prim_cell_prim)  # columns b1,b2,b3 (cartesian)
    q_cart_path = np.dot(frac_path, B.T)

    # compute frequencies along path
    freqs_all = []
    t0 = time.time()
    for q in q_cart_path:
        # 通用 primitive 构造
        Dq = build_D_q_from_supercell_IFC_primitive(
            Phi_super, positions_super, prim_cell_prim,
            basis_fracs_prim, masses_basis_prim, q, tol=1e-3
        )
        freqs = eigfreqs_from_D(Dq)
        freqs_all.append(freqs)
    t1 = time.time()
    print(f"[INFO] phonon frequencies computed in {t1-t0:.1f} s")
    freqs_all = np.array(freqs_all)   # shape (nq, 3*nbasis)

    # 将整条路径拆成三段：Γ->X, K->Γ, Γ->L（每段 npts_seg+1 个点）
    L1 = npts_seg + 1
    L2 = npts_seg + 1
    L3 = npts_seg + 1
    q1 = q_cart_path[0:L1]
    q2 = q_cart_path[L1:L1+L2]
    q3 = q_cart_path[L1+L2:L1+L2+L3]

    def cumdist(Q):
        if len(Q) <= 1:
            return np.array([0.0], dtype=float)
        d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(d)))

    # 分段累计距离，并做连续偏移
    qd1 = cumdist(q1)
    qd2 = cumdist(q2) + qd1[-1]
    qd3 = cumdist(q3) + qd2[-1]

    # 构造带 NaN 的 x 以断开段与段之间的连线
    x_all = np.concatenate([qd1, [np.nan], qd2, [np.nan], qd3])

    plt.figure(figsize=(8,6))
    # 按段绘制（用 NaN 断点避免直线连接）
    for band in range(freqs_all.shape[1]):
        y_all = np.concatenate([
            freqs_all[0:L1, band],
            [np.nan],
            freqs_all[L1:L1+L2, band],
            [np.nan],
            freqs_all[L1+L2:L1+L2+L3, band]
        ])
        plt.plot(x_all, y_all, color='k', lw=1.0)

    # ticks：段终点累计距离
    ticks = [qd1[0], qd1[-1], qd2[-1], qd3[-1]]
    labels_ticks = [r'$\Gamma$', 'X|K', r'$\Gamma$', 'L']
    plt.xticks(ticks, labels_ticks)
    plt.ylabel("Frequency (THz)")
    plt.title("Phonon dispersion-Al (primitive)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)

if __name__ == "__main__":

    # === 参数设置 ===
    main(
        use_diamond = False,     # False -> Al fcc example; True -> diamond (需要对应势)
        delta = 1e-3 ,          # Å finite difference in calculating Phi
        nx=2,ny=2,nz=2,       # 超胞尺寸
        npts_seg = 50 ,         # 每段 q 点数
        use_primitive_2 = True,
        filepath = "./test_al.png"
        )
