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
    - calc.energy_and_forces(positions, cell) 返回 (E, forces)
    - positions_super: (N_super,3)
    - cell_super: 3x3 cell for supercell (cartesian lattice vectors as columns)
    - nbasis: number of atoms in primitive cell (e.g., 1 for simple cubic, 2 for diamond primitive)
    - nx,ny,nz: supercell replication factors
    返回:
      Phi_prim: dict mapping (i_basis, j_basis, shift_tuple) -> 3x3 force constant block (in eV/Å^2)
      R_list: list of unique shift tuples
    """
    N_super = len(positions_super)
    mapping = build_supercell_mapping(nbasis, nx, ny, nz)  # length N_super
    # compute primitive cell (cartesian lattice vectors) by dividing supercell cell by nx,ny,nz
    # assume cell_super is diagonal or general: primitive_cell = cell_super * diag(1/nx,1/ny,1/nz)
    prim_cell = cell_super.copy()
    # divide columns by nx,ny,nz respectively
    prim_cell[:,0] /= nx
    prim_cell[:,1] /= ny
    prim_cell[:,2] /= nz

    # arrays to store supercell Phi: shape (N_super,3,N_super,3)
    Phi_super = np.zeros((N_super,3,N_super,3), dtype=float)

    print(f"[INFO] Supercell size: {nx}x{ny}x{nz}, N_super = {N_super}")
    t0 = time.time()
    # finite displacement for each supercell atom j and direction beta
    for j in range(N_super):
        for beta in range(3):
            disp = np.zeros_like(positions_super)
            disp[j, beta] = delta
            E_plus, F_plus = calc.energy_and_forces(positions_super + disp, cell_super)
            E_minus, F_minus = calc.energy_and_forces(positions_super - disp, cell_super)
            dF = (F_plus - F_minus) / (2.0 * delta)
            # negative derivative: Phi(iα, jβ) = - ∂F_iα / ∂u_jβ
            for i in range(N_super):
                Phi_super[i, :, j, beta] = -dF[i, :]
        # progress info
        if (j+1) % 5 == 0 or j==N_super-1:
            print(f"  computed displacements for atom {j+1}/{N_super}")
    t1 = time.time()
    print(f"[INFO] Finite difference done in {t1-t0:.1f} s")

    # fold supercell Phi into primitive-cell-centered IFCs:
    # Phi_prim[(ib, jb, shift_tuple)] = 3x3 block
    Phi_prim = {}  # dict to accumulate blocks
    R_set = set()

    for p in range(N_super):
        ib, shift_p = mapping[p]
        for q in range(N_super):
            jb, shift_q = mapping[q]
            # R = shift_q - shift_p (vector of integers)
            R_shift = (shift_q[0] - shift_p[0], shift_q[1] - shift_p[1], shift_q[2] - shift_p[2])
            R_set.add(R_shift)
            key = (ib, jb, R_shift)
            block = Phi_super[p, :, q, :]  # 3x3
            # accumulate (there should be exactly one contributing pair per mapping, but sum anyway)
            if key in Phi_prim:
                Phi_prim[key] += block
            else:
                Phi_prim[key] = block.copy()

    # convert R_set to sorted list (for reproducibility)
    R_list = sorted(list(R_set))
    print(f"[INFO] Found {len(R_list)} translation vectors (R) in supercell")
    return Phi_prim, R_list, prim_cell

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
    # fractional coords in conventional cubic primitive reciprocal basis
    G = np.array([0.0, 0.0, 0.0])
    X = np.array([0.5, 0.0, 0.5])  # depends on chosen basis; for fcc conv cell this is common
    L = np.array([0.5, 0.5, 0.5])
    path = []
    labels = [r'$\Gamma$', 'X', 'L', r'$\Gamma$']
    segs = [(G,X),(X,L),(L,G)]
    for start,end in segs:
        for t in np.linspace(0,1,nseg,endpoint=False):
            path.append((1-t)*start + t*end)
    path.append(G)
    return np.array(path), labels

# -------------------------
# 5) 主流程示例（2x2x2 超胞）
# -------------------------
if __name__ == "__main__":

    from potential import DirectLAMMPSEAMself as LMPCalc  # adapt if class name different
    from opt_method import make_fcc, make_diamond, build_supercell

    # === 参数设置 ===
    use_diamond = False   # False -> Al fcc example; True -> diamond (需要对应势)
    delta = 1e-3          # Å
    nx,ny,nz = 2,2,2      # 超胞尺寸
    npts_seg = 20         # 每段 q 点数

    if not use_diamond:
        # Al example (EAM)
        calc = LMPCalc(
            eam_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\Al_zhou.eam.alloy",
            lmp_cmd=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
            element="Al",
            pair_style="eam/alloy",
            mass=26.981538,
            keep_tmp_files=False
        )
        lat = make_fcc(4.05)
    else:
        calc = LMPCalc(
            eam_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
            lmp_cmd=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
            element="C",
            pair_style="lcbop",
            mass=12.011,
            keep_tmp_files=False
        )
        lat = make_diamond(3.567)

    # build supercell positions and supercell matrix
    positions_super, cell_super, symbols_super = build_supercell(lat, nx, ny, nz)
    N_super = len(positions_super)
    nbasis = len(lat.basis_frac)  # atoms per primitive cell
    print(f"Primitive basis atoms: {nbasis}, Supercell atoms: {N_super}")

    # compute force constants on supercell
    Phi_prim_dict, R_list, prim_cell = compute_force_constants_supercell(
        calc, positions_super, cell_super, nbasis, nx,ny,nz, delta=delta
    )

    # masses per basis (amu)
    if not use_diamond:
        masses_basis = np.array([26.981538] * nbasis)
    else:
        masses_basis = np.array([12.011] * nbasis)

    # prepare q path in cartesian coords (1/Å)
    frac_path, labels = path_G_X_L_G(nseg=npts_seg)
    B = reciprocal_vectors(prim_cell)  # columns b1,b2,b3 (cartesian)
    q_cart_path = np.dot(frac_path, B.T)  # each frac in basis of b1,b2,b3 ; result in 1/Å

    # compute frequencies along path
    freqs_all = []
    t0 = time.time()
    for q in q_cart_path:
        Dq = build_D_q_from_prim_IFC(Phi_prim_dict, R_list, prim_cell, masses_basis, q)
        freqs = eigfreqs_from_D(Dq)
        freqs_all.append(freqs)
    t1 = time.time()
    print(f"[INFO] phonon frequencies computed in {t1-t0:.1f} s")

    freqs_all = np.array(freqs_all)   # shape (nq, 3*nbasis)

    # plot
    # x coordinates: cumulative distance along fractional path in reciprocal space
    dq = np.linalg.norm(np.diff(q_cart_path, axis=0), axis=1)
    qdist = np.concatenate(([0.0], np.cumsum(dq)))
    plt.figure(figsize=(8,6))
    for band in range(freqs_all.shape[1]):
        plt.plot(qdist, freqs_all[:, band], color='k')
    # ticks at segment boundaries
    seg1 = npts_seg
    seg2 = 2 * npts_seg
    ticks = [qdist[0], qdist[seg1], qdist[seg2], qdist[-1]]
    plt.xticks(ticks, labels)
    plt.ylabel("Frequency (THz)")
    plt.title("Phonon dispersion (finite displacement, 2x2x2 supercell)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("./phonon_eam_0K.png")
    plt.show()
