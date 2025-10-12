"""
structure_optimization_module.py
模块 1: 结构优化 (0 K) 模块

- 构建晶胞/超胞 fcc, diamond
- LJ 占位势（可替换）
- Verlet neighbor list 可选
- FIRE 最小化（原子坐标松弛）
- 标量晶格常数搜索（对 a 做 1D 扫描并在每点做原子弛豫）
- Virial stress 势能项 计算 (0 K)

注意：
- 所有长度单位为 Å，能量单位为 eV(占位势参数也使用这些单位)
- 该模块仅实现经典原子模拟层: EAM/Tersoff 需在 LJPotential 接口替换或实现新类
"""

import numpy as np
import itertools

# ------------------------
# 基本结构定义与构建
# ------------------------
class Lattice:
    def __init__(self, name, a, basis_frac, symbols):
        """
        name: 字符串
        a: 格常 (Å)
        basis_frac: (n_basis,3) 分数坐标
        symbols: list of atom symbols length n_basis
        """
        self.name = name
        self.a = a
        self.basis_frac = np.array(basis_frac, dtype=float)
        self.symbols = symbols

def make_fcc(a):
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    syms = ['Al'] * 4
    return Lattice('fcc', a, basis, syms)

def make_diamond(a):
    """
    Diamond 结构的常规立方晶胞（simple cubic Bravais）+ 8 原子基矢。
    注意：不要用 sc Bravais + 2 原子 (0,0,0) 与 (1/4,1/4,1/4)，那是 fcc Bravais 的表达法。
    这里用 8 原子常规胞，便于与当前 build_supercell(正交对角 cell) 兼容。
    """
    basis = np.array([
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [0.00, 0.50, 0.50],
        [0.25, 0.75, 0.75],
        [0.50, 0.00, 0.50],
        [0.75, 0.25, 0.75],
        [0.50, 0.50, 0.00],
        [0.75, 0.75, 0.25],
    ])
    syms = ['C'] * len(basis)
    return Lattice('diamond', a, basis, syms)

def build_supercell(lattice, nx, ny, nz):
    """
    return:
      positions: (N,3) Cartesian coordinates in Å (within cell)
      cell: 3x3 matrix, orthogonal diag([nx*a, ny*a, nz*a])
      symbols: list length N
    """
    a = lattice.a
    coords = []
    syms = []
    for ix,iy,iz in itertools.product(range(nx), range(ny), range(nz)):
        shift = np.array([ix,iy,iz], dtype=float)
        for i, frac in enumerate(lattice.basis_frac):
            r = (frac + shift) * a
            coords.append(r)
            syms.append(lattice.symbols[i])
    coords = np.array(coords)
    cell = np.diag([nx*a, ny*a, nz*a])
    return coords, cell, syms

# ------------------------
# PBC and  minimum image
# ------------------------
def min_image(dr, cell, invcell):
    """
    dr: cartesian difference r_i - r_j
    cell: 3x3
    invcell: inverse of cell
    returns: wrapped dr (cartesian) using minimum image via fractional coords
    """
    frac = invcell @ dr
    # wrap to (-0.5, 0.5]
    frac -= np.floor(frac + 0.5)
    return cell @ frac

def wrap_positions(positions, cell):
    """
    Wrap Cartesian positions back into the unit cell.
    positions: (N,3)
    """
    invcell = np.linalg.inv(cell)
    fracs = (invcell @ positions.T).T
    fracs -= np.floor(fracs)
    return (cell @ fracs.T).T

# ------------------------
# Neighbor list (Verlet)
# ------------------------
class VerletNeighborList:
    def __init__(self, cutoff, skin=0.3):
        """
        cutoff: neighbor cutoff distance (Å)
        skin: extra margin for neighbor list (Å)
        """
        self.cutoff = cutoff
        self.skin = skin
        self.build_cutoff = cutoff + skin
        self.list = None
        self.positions_hash = None  # simple invalidation mechanism

    def build(self, positions, cell):
        N = len(positions)
        invcell = np.linalg.inv(cell)
        build_rc2 = (self.build_cutoff)**2
        self.list = [[] for _ in range(N)]
        # naive O(N^2) build (sufficient for small systems)
        for i in range(N-1):
            ri = positions[i]
            for j in range(i+1, N):
                rj = positions[j]
                dr = ri - rj
                dr = min_image(dr, cell, invcell)
                r2 = dr.dot(dr)
                if r2 <= build_rc2:
                    self.list[i].append(j)
                    self.list[j].append(i)
        # store small hash (shape + cell diag) to maybe detect changes
        self.positions_hash = (positions.shape, tuple(np.diag(cell)))
        return self.list

    def neighbors_of(self, i):
        return self.list[i]

# ------------------------
# 共轭梯度优化器（0K，固定晶胞，仅原子坐标）
# ------------------------
def cg_minimize(positions, cell, potential, neighbor_list=None,
                f_tol=1e-6, maxit=5000, ls_c1=1e-4, ls_shrink=0.5,
                step_init=0.1, verbose=False):
    """
    非线性共轭梯度 (Fletcher-Reeves + Armijo 回溯线搜索)，用于0K下固定晶胞的原子坐标优化。
    参数:
      positions: (N,3) Å
      cell: (3,3) Å
      potential: PotentialBase
      neighbor_list: VerletNeighborList 或 None（注意：此实现不自动重建邻居表）
      f_tol: 收敛阈值，max |F_i| < f_tol 视为收敛
      maxit: 最大迭代步
      ls_c1: Armijo 条件常数
      ls_shrink: 线搜索收缩因子
      step_init: 初始步长（Å）
    返回: positions_relaxed, energy, forces, iterations
    """
    x = positions.copy()
    E, F = potential.energy_and_forces(x, cell, neighbor_list)
    g = -F  # 以能量梯度 g = -F
    d = -g  # 初始下降方向
    step = step_init

    for it in range(1, maxit+1):
        fmax = np.max(np.linalg.norm(F, axis=1))
        if fmax < f_tol:
            if verbose:
                print(f"[CG] converged at it={it}, fmax={fmax:.3e}, E={E:.6e}")
            return x, E, F, it

        # Armijo 回溯线搜索
        gdotd = float(np.sum(g * d))
        if gdotd >= 0:
            # 方向非下降，重置为最速下降
            d = -g
            gdotd = float(np.sum(g * d))

        t = step
        accepted = False
        for _ in range(50):
            x_trial = x + t * d
            x_trial = wrap_positions(x_trial, cell)
            E_trial, F_trial = potential.energy_and_forces(x_trial, cell, neighbor_list)
            if E_trial <= E + ls_c1 * t * gdotd:
                accepted = True
                break
            t *= ls_shrink

        if not accepted:
            # 如果线搜索失败，退化为很小步长的最速下降
            t = step * 1e-3
            x_trial = x + t * (-g)
            x_trial = wrap_positions(x_trial, cell)
            E_trial, F_trial = potential.energy_and_forces(x_trial, cell, neighbor_list)

        # 接受步
        x, E, F = x_trial, E_trial, F_trial
        g_new = -F

        # Fletcher-Reeves beta
        num = np.sum(g_new*g_new)
        den = np.sum(g*g) + 1e-30
        beta = max(0.0, num / den)

        d = -g_new + beta * d
        g = g_new

        # 简单步长调节
        if accepted:
            step = min(t * 1.2, 1.0)
        else:
            step = max(step * 0.5, 1e-4)

        if verbose and (it % 50 == 0):
            print(f"[CG] it={it} E={E:.6e} fmax={np.max(np.linalg.norm(F,axis=1)):.3e} step={t:.3e}")

    if verbose:
        print(f"[CG] not converged after {maxit} steps, fmax={np.max(np.linalg.norm(F,axis=1)):.3e}")
    return x, E, F, maxit

# ------------------------
# 施加小应变的仿射变换
# ------------------------
def apply_strain(cell, positions, eta):
    """
    给定小应变张量 eta（3x3），构造 F = I + eta。
    返回: new_cell, positions_affine(作为新结构的初始猜测)
    注：positions 为笛卡尔坐标，仿射变换 r' = F r；再包裹进新胞。
    """
    F = np.eye(3) + np.array(eta, dtype=float)
    new_cell = F @ cell
    pos_aff = (F @ positions.T).T
    pos_aff = wrap_positions(pos_aff, new_cell)
    return new_cell, pos_aff

# ------------------------
# 固定晶胞的松弛  CG
# ------------------------
def relax_positions_fixed_cell(positions, cell, potential,
                               use_nlist=False, cutoff=None, skin=0.3,
                               f_tol=1e-6, maxit=5000, verbose=False,
                               method='cg'):
    """
    固定晶胞下的内部坐标优化。
    method: 'cg' 'RMM-DIIS'
    method:
    - 'cg': 使用本模块内置共轭梯度 (需要 potential.energy_and_forces)
    - 'external': 调用 potential.relax_positions_fixed_cell (例如 DirectLAMMPSEAMPotential)
    返回: positions_relaxed, energy, forces, iter_count, neighbor_list_used
    """
    # 外接 LAMMPS 或其它后端：优先走 external
    if method == 'external' and hasattr(potential, 'relax_positions_fixed_cell'):
        # 将本接口的容差/迭代数映射到外部接口
        etol   = 1e-12
        ftol   = f_tol
        maxiter= int(maxit)
        maxeval= max(10*int(maxit), 10000)
        R_relaxed, E = potential.relax_positions_fixed_cell(
            positions, cell,
            min_style="fire", e_tol=etol, f_tol=ftol,
            maxiter=maxiter, maxeval=maxeval,
            align_to_input=True
        )
        F_dummy = np.zeros_like(positions)
        it_dummy = -1
        return R_relaxed, E, F_dummy, it_dummy, None
    
    neighbor_list = None
    if use_nlist:
        if cutoff is None:
            raise ValueError("cutoff must be provided when use_nlist=True")
        neighbor_list = VerletNeighborList(cutoff=cutoff, skin=skin)
        neighbor_list.build(positions, cell)

    if method == 'cg':
        pos_relaxed, E, F, it = cg_minimize(positions, cell, potential,
                                            neighbor_list=neighbor_list,
                                            f_tol=f_tol, maxit=maxit, verbose=verbose)
    else:
        raise ValueError(f"unknown method '{method}', expected 'cg' or 'external'")
    return pos_relaxed, E, F, it, neighbor_list

# ------------------------
# 标量晶格常数 a 的 0K 优化（一维黄金分割搜索）
# ------------------------
def optimize_scalar_a_0K(lattice_factory, a_init, nx, ny, nz, potential,
                         relax_params=None, use_nlist=False, cutoff=None,
                         bracket_frac=0.06, tol_rel=1e-4, max_iter=30, verbose=False,
                         method='cg'):
    """
    在 a 的一维空间做 0K 优化：对每个 a，重建超胞并在固定晶胞下松弛原子坐标，最小化总能量。
    返回: a_opt, pos0_opt, cell0_opt, E0_opt
    """
    if relax_params is None:
        relax_params = {}

    def eval_at_a(a):
        pos, cell, _ = build_supercell(lattice_factory(a), nx, ny, nz)
        pos_rel, E, _, _, _ = relax_positions_fixed_cell(
            pos, cell, potential,
            use_nlist=use_nlist, cutoff=cutoff, verbose=False, method=method,**relax_params
        )
        return E, pos_rel, cell

    # 初始区间 [aL, aR]
    a0 = float(a_init)
    aL = a0 * (1.0 - bracket_frac)
    aR = a0 * (1.0 + bracket_frac)
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618...

    # 内点
    c = aR - gr * (aR - aL)
    d = aL + gr * (aR - aL)

    Ec, pos_c, cell_c = eval_at_a(c)
    Ed, pos_d, cell_d = eval_at_a(d)
    best = {'a': c if Ec < Ed else d,
            'E': Ec if Ec < Ed else Ed,
            'pos': pos_c if Ec < Ed else pos_d,
            'cell': cell_c if Ec < Ed else cell_d}

    it = 0
    while it < max_iter and (aR - aL) / max(a0, 1e-8) > tol_rel:
        if Ec < Ed:
            aR = d
            d = c
            Ed = Ec
            pos_d, cell_d = pos_c, cell_c
            c = aR - gr * (aR - aL)
            Ec, pos_c, cell_c = eval_at_a(c)
            if Ec < best['E']:
                best = {'a': c, 'E': Ec, 'pos': pos_c, 'cell': cell_c}
        else:
            aL = c
            c = d
            Ec = Ed
            pos_c, cell_c = pos_d, cell_d
            d = aL + gr * (aR - aL)
            Ed, pos_d, cell_d = eval_at_a(d)
            if Ed < best['E']:
                best = {'a': d, 'E': Ed, 'pos': pos_d, 'cell': cell_d}
        it += 1
        if verbose:
            print(f"[opt-a] it={it} interval=({aL:.6f},{aR:.6f}) bestE={best['E']:.6e} a*={best['a']:.6f}")
 
 # 取区间中心再评估一次，防止错过端点附近更优值
    amid = 0.5 * (aL + aR)
    Em, pos_m, cell_m = eval_at_a(amid)
    if Em < best['E']:
        best = {'a': amid, 'E': Em, 'pos': pos_m, 'cell': cell_m}

    if verbose:
        print(f"[opt-a] done: a_opt={best['a']:.6f}, E0={best['E']:.6e}")
    return best['a'], best['pos'], best['cell'], best['E']

# ------------------------
# 0K 下的应变-应力（有限差分能量导数）
# ------------------------
def stress_via_energy_fd_0K(positions, cell, potential,
                            strain_eps=1e-4, symmetric=True,
                            relax_params=None, use_nlist=False, cutoff=None,
                            volume_ref='reference', verbose=False, method='cg'):
    """
    使用中心差分对能量对应变的导数计算应力: sigma_ij =  (1/V_ref) * ∂E/∂η_ji
    参数:
      positions, cell: 初始结构（将先在未应变胞下做基态松弛）
      strain_eps: 单个分量的应变扰动幅度
      symmetric: True 表示对剪切用对称应变(η_ij=η_ji)，False 则只扰动单个分量
      volume_ref: 'reference' 用初始未应变体积 V0，'current' 用每次应变后的体积
      method: 'cg' 或 'fire'，用于固定晶胞下的坐标优化
    返回:
      sigma: (3,3) eV/Å^3
      E0: 基态能量（未应变）
      pos0, cell0: 基态构型
    """
    if relax_params is None:
        relax_params = {}
    # 1) 基态松弛（未应变）
    pos0_rel, E0, _, _, _ = relax_positions_fixed_cell(
        positions, cell, potential,
        use_nlist=use_nlist, cutoff=cutoff,
        verbose=verbose, method=method, **relax_params
    )
    V0 = float(np.linalg.det(cell))
    sigma = np.zeros((3,3))

    # 2) 对每个分量做中心差分
    for i in range(3):
        for j in range(3):
            # 构造 ± 扰动的应变张量
            eta_plus = np.zeros((3,3))
            eta_minus = np.zeros((3,3))
            eta_plus[i, j] = +strain_eps
            eta_minus[i, j] = -strain_eps
            if symmetric and i != j:
                eta_plus[j, i] = +strain_eps
                eta_minus[j, i] = -strain_eps

            # +strain
            cell_p, pos_p0 = apply_strain(cell, pos0_rel, eta_plus)
            pos_p, Ep, Fp, _, _ = relax_positions_fixed_cell(
                pos_p0, cell_p, potential,
                use_nlist=use_nlist, cutoff=cutoff,
                verbose=False, method=method, **relax_params
            )
            Vp = float(np.linalg.det(cell_p))

            # -strain
            cell_m, pos_m0 = apply_strain(cell, pos0_rel, eta_minus)
            pos_m, Em, _, _, _ = relax_positions_fixed_cell(
                pos_m0, cell_m, potential,
                use_nlist=use_nlist, cutoff=cutoff,
                verbose=False, method=method, **relax_params
            )
            Vm = float(np.linalg.det(cell_m))

            dE_deta_ij = (Ep - Em) / (2.0 * strain_eps)
            if volume_ref == 'current':
                Vref = 0.5 * (Vp + Vm)
            else:
                Vref = V0
            # 用户给定公式：σ_ij = -∂E/∂η_ji / Vref
            sigma[i, j] =  dE_deta_ij / Vref

    return sigma, E0, pos0_rel, cell

# ------------------------
# 示例：0K 下的应变-应力计算入口
# ------------------------

def strain_stress_0K_pipeline(lattice_factory, a, nx, ny, nz, potential,
                              strain_eps=1e-4, symmetric=True,
                              relax_params=None, use_nlist=False, cutoff=None,
                              optimize_a=True, bracket_frac=0.06,
                              verbose=False, method='cg'):    
    """
    构建超胞 -> 基态松弛 -> 有限差分应变-应力
    返回: dict {sigma, E0, pos0, cell0}
    """
    if optimize_a:
        a_opt, pos0, cell0, E0 = optimize_scalar_a_0K(
            lattice_factory, a, nx, ny, nz, potential,
            relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
            bracket_frac=bracket_frac, verbose=verbose, method=method,
        )
    else:
        a_opt = a
        pos0, cell0, _ = build_supercell(lattice_factory(a_opt), nx, ny, nz)
        pos0, E0, _, _, _ = relax_positions_fixed_cell(
            pos0, cell0, potential,
            use_nlist=use_nlist, cutoff=cutoff,
            verbose=verbose, method=method,**(relax_params or {})
        )

    sigma, _, _, _ = stress_via_energy_fd_0K(
        pos0, cell0, potential,
        strain_eps=strain_eps, symmetric=symmetric,
        relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
        method=method, verbose=verbose
    )
    return {'sigma': sigma, 'E0': E0, 'pos0': pos0, 'cell0': cell0,'a_opt': a_opt}

# ------------------------
# 单分量应变-应力（中心差分，0K 固定晶胞）
# ------------------------
def stress_component_fd_at_strain(pos0_rel, cell, potential, i, j, eps0, delta=1e-4,
                                  symmetric=True, relax_params=None,
                                  use_nlist=False, cutoff=None,
                                  volume_ref='reference', verbose=False,
                                  method='cg'):
    """
    计算 σ_ij(ε0)，使用中心差分：σ_ij(ε0) = (1/V_ref) * [E(ε0+δ) - E(ε0-δ)] / (2δ)
    其中 δ>0。
    采用能量对应变的中心差分，0K 固定晶胞下每点重松弛原子。
    参数:
      pos0_rel, cell: 未应变基态构型（已在 cell 下松弛）
      i, j: 分量索引 0..2，对应 x,y,z
      strain_eps: 扰动幅度（标量）
      symmetric: True 时 i!=j 采用对称剪切 η_ij=η_ji=ε
      volume_ref: 'reference' 使用未应变体积，'current' 使用两次应变体积平均
    返回:
      sigma_ij: eV/Å^3
    说明:
      按你给定公式使用 σ_ij = (1/V_ref) * ∂E/∂η_ji。为简洁按矩阵同位赋值，与上面的整矩阵函数一致。
    """
    if relax_params is None:
        relax_params = {}

    V0 = float(np.linalg.det(cell))
    delta = abs(float(delta))

    # ε0 + δ
    eta_p = np.zeros((3,3))
    eta_p[i, j] = eps0 + delta
    if symmetric and i != j:
        eta_p[j, i] = eps0 + delta
    cell_p, pos_p0 = apply_strain(cell, pos0_rel, eta_p)
    _, Ep, _, _, _ = relax_positions_fixed_cell(
        pos_p0, cell_p, potential,
        use_nlist=use_nlist, cutoff=cutoff,
        verbose=False, method=method,  **relax_params
    )
    Vp = float(np.linalg.det(cell_p))
    # ε0 - δ
    eta_m = np.zeros((3,3))
    eta_m[i, j] = eps0 - delta
    if symmetric and i != j:
        eta_m[j, i] = eps0 - delta
    cell_m, pos_m0 = apply_strain(cell, pos0_rel, eta_m)
    _, Em, _, _, _ = relax_positions_fixed_cell(
        pos_m0, cell_m, potential,
        use_nlist=use_nlist, cutoff=cutoff,
        verbose=False, method=method,  **relax_params
    )
    Vm = float(np.linalg.det(cell_m))

    dE = (Ep - Em) / (2.0 * delta)
    Vref = 0.5*(Vp+Vm) if volume_ref == 'current' else V0
    sigma_ij = dE / Vref
    return sigma_ij

# ------------------------
# 扫描 σ_ij(ε) 并可保存与绘图
# ------------------------
def scan_sigma_component_vs_strain(pos0_rel, cell, potential, i, j, eps_list,
                                   delta_fd=1e-4, symmetric=True, relax_params=None,
                                   use_nlist=False, cutoff=None,
                                   volume_ref='reference', verbose=False,
                                   save_csv_path=None,method='cg'):
    """
    扫描 σ_ij(ε)，对每个 ε0 用中心差分 δ=delta_fd>0。
    """
    eps_arr = np.asarray(eps_list, dtype=float)
    sig = []
    for eps0 in eps_arr:
        sij = stress_component_fd_at_strain(pos0_rel, cell, potential, i, j, eps0,
                                            delta=delta_fd, symmetric=symmetric,
                                            relax_params=relax_params,
                                            use_nlist=use_nlist, cutoff=cutoff,
                                            volume_ref=volume_ref, verbose=verbose,
                                            method=method)
        sig.append(sij)
    sigma_eVA3 = np.array(sig, dtype=float)
    sigma_GPa = sigma_eVA3 * 160.21766208

    if save_csv_path is not None:
        import os, csv
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        with open(save_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epsilon0", f"sigma_{i}{j}_eV_A3", f"sigma_{i}{j}_GPa"])
            for e, s1, s2 in zip(eps_arr, sigma_eVA3, sigma_GPa):
                w.writerow([f"{e:.8e}", f"{s1:.8e}", f"{s2:.8e}"])
    return eps_arr, sigma_eVA3, sigma_GPa

def plot_sigma_components_vs_strain(eps_arr, sigma_dict, unit="GPa", title=None, save_png=None):
    """
    sigma_dict: { (i,j): sigma_vals_eV_A3(ndarray 与 eps_arr同长), ... }
    unit: 'GPa' 或 'eV/A^3'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图。请先 pip install matplotlib")
        return
    factor = 160.21766208 if unit.upper()=="GPA" else 1.0
    plt.figure()
    for (i, j), sigma_vals in sigma_dict.items():
        y = sigma_vals * factor
        # 标签用 1-based 记号更直观：σ11, σ12, σ23 ...
        plt.plot(eps_arr, y, marker='o', lw=1.8, ms=4, label=f"σ{i+1}{j+1}")
    plt.xlabel("strain ε")
    plt.ylabel(f"σ_ij ({unit})")
    if title:
        plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    if save_png:
        plt.savefig(save_png, dpi=220, bbox_inches="tight")


def scan_sigma_component_vs_strain_virial(pos0_rel, cell, potential, i, j, eps_list,
                                          symmetric=True, relax_params=None, verbose=False):
    """
    使用 LAMMPS 直接返回的整体 virial 压力张量（换算为应力），避免能量差分带来的偏置。
    要求 potential 提供 relax_energy_and_stress(positions, cell) 接口。
    """
    if relax_params is None:
        relax_params = {}
    eps_arr = np.asarray(eps_list, dtype=float)
    sig = []
    for eps0 in eps_arr:
        eta = np.zeros((3,3))
        eta[i, j] = eps0
        if symmetric and i != j:
            eta[j, i] = eps0
        cell_s, pos_s0 = apply_strain(cell, pos0_rel, eta)
        if not hasattr(potential, "relax_energy_and_stress"):
            raise RuntimeError("potential 未实现 relax_energy_and_stress()")
        _, _, sigma = potential.relax_energy_and_stress(
            pos_s0, cell_s,
            min_style="fire",
            e_tol=relax_params.get("e_tol", 1e-12),
            f_tol=relax_params.get("f_tol", 1e-6),
            maxiter=relax_params.get("maxit", 20000),
            maxeval=max(10*relax_params.get("maxit", 20000), 200000),
            align_to_input=True
        )
        sig.append(float(sigma[i, j]))
    sigma_eVA3 = np.array(sig, dtype=float)
    sigma_GPa = sigma_eVA3 * 160.21766208
    return eps_arr, sigma_eVA3, sigma_GPa


# ------------------------
# Example usage (do not run in module import if you don't want automatic run)
# ------------------------
if __name__ == "__main__":
    # Quick demo parameters (LJ placeholders)
    #from potential import LJPotential
    #al_lat = make_fcc(a=4.05)  # 初始猜测
    #nx,ny,nz = 2,2,2
    #lj = LJPotential(eps=0.0103, sigma=2.5)
    #out = strain_stress_0K_pipeline(
    #    lattice_factory=make_fcc, a=al_lat.a, nx=nx, ny=ny, nz=nz, potential=lj,
    #    strain_eps=1e-4, optimize_a=True, verbose=True,
    #    relax_params=dict(f_tol=1e-6, maxit=2000)
    #)
    #print("a_opt =", out['a_opt'])
    #print("sigma (eV/Å^3) =\n", out['sigma'])

    # 扫描示例：σ_xx 对不同应变幅度 ε 的关系（在 a_opt 基态上）
    #try:
    #    eps_list = np.linspace(-9e-3, 9e-3, 20)  # 建议对称区间，便于观察线性区
    #    comps = [(0,0),(1,1), (2,2), (1,2)]  # 11, 33, 23
    #    sigma_dict = {}
    #    eps_arr_ref = None        
    #    for (i, j) in comps:
    #        eps_arr, sigma_eVA3, _ = scan_sigma_component_vs_strain(
    #            out['pos0'], out['cell0'], lj, i, j, eps_list,
    #            delta_fd=1e-4, symmetric=True, relax_params=dict(f_tol=1e-6, maxit=2000),use_nlist=False, cutoff=None,
    #            volume_ref='reference', verbose=False,
    #            save_csv_path=None
    #        )
    #        sigma_dict[(i, j)] = sigma_eVA3
    #        if eps_arr_ref is None:
    #            eps_arr_ref = eps_arr
    #    plot_sigma_components_vs_strain(
    #        eps_arr_ref, sigma_dict, unit="GPa",
    #        title=None, save_png="./sigma_multi_scan.png"
    #    )
    #except Exception as e:
    #    print("扫描/绘图示例出错：", e)
    """
    from potential import DirectLAMMPSEAMPotential
    nx, ny, nz = 2, 2, 2
    a_guess = 4.00
    pot = DirectLAMMPSEAMPotential(
        eam_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\Al_zhou.eam.alloy",
        lmp_cmd =r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
        element="Al", pair_style="eam/alloy", keep_tmp_files=False
    )
    out = strain_stress_0K_pipeline(
        lattice_factory=make_fcc, a=a_guess, nx=nx, ny=ny, nz=nz, potential=pot,
        strain_eps=1e-3, optimize_a=True, verbose=True,
        relax_params=dict(f_tol=1e-6, maxit=2000),  # 将会映射到 etol/ftol/maxiter
        method='external'
    )
    # 扫描多分量对比
    import numpy as np
    eps_list = np.linspace(-8e-3, 8e-3, 16)
    comps = [(0,0), (0,1), (1,2)]
    sigma_dict = {}
    for (i, j) in comps:
        eps_arr, sigma_eVA3, _ = scan_sigma_component_vs_strain(
            out['pos0'], out['cell0'], pot, i, j, eps_list,
            delta_fd=2e-4, symmetric=True, relax_params=dict(f_tol=1e-6, maxit=2000),
            use_nlist=False, cutoff=None, volume_ref='reference', verbose=False,
            save_csv_path=None, method='external'
        )
        sigma_dict[(i, j)] = sigma_eVA3
    plot_sigma_components_vs_strain(eps_arr, sigma_dict, unit="GPa",
                                    title=None, save_png="./test_Al_eam.png")
    
    
    from potential import DirectLAMMPSLCBOPPotential
    nx, ny, nz = 2, 2, 2
    a_guess = 3.6
    pot = DirectLAMMPSLCBOPPotential(
        lcbop_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
        lmp_cmd =r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
        element="C", pair_style="lcbop", keep_tmp_files=False
    )
    out = strain_stress_0K_pipeline(
        lattice_factory=make_diamond, a=a_guess, nx=nx, ny=ny, nz=nz, potential=pot,
        strain_eps=1e-3, optimize_a=True, verbose=True,
        relax_params=dict(f_tol=1e-6, maxit=2000),  # 将会映射到 etol/ftol/maxiter
        method='external'
    )

    eps_list = np.linspace(-8e-3, 8e-3, 16)
    comps = [(0,0),(0,1)]
    sigma_dict = {}
    for (i, j) in comps:
        eps_arr, sigma_eVA3, _ = scan_sigma_component_vs_strain(
            out['pos0'], out['cell0'], pot, i, j, eps_list,
            delta_fd=2e-4, symmetric=True, relax_params=dict(f_tol=1e-6, maxit=2000),
            use_nlist=False, cutoff=None, volume_ref='reference', verbose=False,
            save_csv_path=None, method='external'
        )
        sigma_dict[(i, j)] = sigma_eVA3
    plot_sigma_components_vs_strain(eps_arr, sigma_dict, unit="GPa",
                                    title=None, save_png="./sigma_multi_C_lcbop.png")
    """

    from potential import DirectLAMMPSLCBOPPotential
    nx, ny, nz = 2, 2, 2
    a_guess = 3.6
    pot = DirectLAMMPSLCBOPPotential(
        lcbop_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
        lmp_cmd =r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
        element="C", pair_style="lcbop", keep_tmp_files=False
    )
    out = strain_stress_0K_pipeline(
        lattice_factory=make_diamond, a=a_guess, nx=nx, ny=ny, nz=nz, potential=pot,
        strain_eps=1e-3, optimize_a=True, verbose=True,
        relax_params=dict(f_tol=1e-6, maxit=2000),  # 将会映射到 etol/ftol/maxiter
        method='external'
    )
    eps_list = np.linspace(-8e-3, 8e-3, 16)
    comps = [(0,0), (0,1), (1,2)]
    sigma_dict = {}
    for (i, j) in comps:
        eps_arr, sigma_eVA3, _ = scan_sigma_component_vs_strain(
            out['pos0'], out['cell0'], pot, i, j, eps_list,
            delta_fd=2e-4, symmetric=True, relax_params=dict(f_tol=1e-6, maxit=2000),
            use_nlist=False, cutoff=None, volume_ref='reference', verbose=False,
            save_csv_path=None, method='external'
        )
        sigma_dict[(i, j)] = sigma_eVA3
    plot_sigma_components_vs_strain(eps_arr, sigma_dict, unit="GPa",
                                    title=None, save_png="./sigma_multi_C_lcbop.png")