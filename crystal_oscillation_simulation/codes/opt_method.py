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
    # conventional cubic cell: 2-atom basis on fcc lattice
    basis = np.array([[0,0,0],[0.25,0.25,0.25]])
    syms = ['C'] * 2
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
# Potential interface & LJ implement
# ------------------------
class PotentialBase:
    def energy_and_forces(self, positions, cell, neighbor_list=None):
        """
        返回 (total_energy, forces_array)
        forces shape (N,3)
        """
        raise NotImplementedError

class LJPotential(PotentialBase):
    def __init__(self, eps, sigma, rc=None):
        self.eps = eps
        self.sigma = sigma
        self.rc = rc if rc is not None else 2.5*sigma  # radius cutoff
        self.rc2 = self.rc*self.rc
        # shift for continuity at rc
        self.shift = 4*eps*((sigma/self.rc)**12 - (sigma/self.rc)**6)

    def energy_and_forces(self, positions, cell, neighbor_list=None):
        N = len(positions)
        forces = np.zeros_like(positions)
        energy = 0.0
        vol = np.linalg.det(cell)
        invcell = np.linalg.inv(cell)
        sigma6 = self.sigma**6
        rc2 = self.rc2

        if neighbor_list is None:
            # naive O(N^2)
            for i in range(N-1):
                ri = positions[i]
                for j in range(i+1, N):
                    dr = ri - positions[j]
                    dr = min_image(dr, cell, invcell)
                    r2 = dr.dot(dr)
                    if r2 < rc2:
                        invr2 = 1.0 / r2
                        sr6 = sigma6 * (invr2**3)
                        sr12 = sr6*sr6
                        e = 4*self.eps*(sr12 - sr6) - self.shift
                        energy += e
                        fmag = 24*self.eps*invr2*(2*sr12 - sr6)
                        fvec = fmag * dr
                        forces[i] += fvec
                        forces[j] -= fvec
        else:
            # use neighbor list to iterate pairs
            for i in range(N):
                ri = positions[i]
                for j in neighbor_list.neighbors_of(i):
                    if j <= i:  # ensure each pair handled once
                        continue
                    dr = ri - positions[j]
                    dr = min_image(dr, cell, invcell)
                    r2 = dr.dot(dr)
                    if r2 < rc2:
                        invr2 = 1.0 / r2
                        sr6 = sigma6 * (invr2**3)
                        sr12 = sr6*sr6
                        e = 4*self.eps*(sr12 - sr6) - self.shift
                        energy += e
                        fmag = 24*self.eps*invr2*(2*sr12 - sr6)
                        fvec = fmag * dr
                        forces[i] += fvec
                        forces[j] -= fvec
        return energy, forces

# add more potential EAM for Al, Tersoff for C-diamond
# to be continued...


# first do structure optimization at 0K
# the initial stru-optim module 

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
    返回: positions_relaxed, energy, forces, iter_count, neighbor_list_used
    """
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
        pass
    return pos_relaxed, E, F, it, neighbor_list

# ------------------------
# 标量晶格常数 a 的 0K 优化（一维黄金分割搜索）
# ------------------------
def optimize_scalar_a_0K(lattice_factory, a_init, nx, ny, nz, potential,
                         relax_params=None, use_nlist=False, cutoff=None,
                         bracket_frac=0.06, tol_rel=1e-4, max_iter=30, verbose=False):
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
            use_nlist=use_nlist, cutoff=cutoff, verbose=False, **relax_params
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
    使用中心差分对能量对应变的导数计算应力: sigma_ij = - (1/V_ref) * ∂E/∂η_ji
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
            sigma[i, j] = - dE_deta_ij / Vref

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
            bracket_frac=bracket_frac, verbose=verbose
        )
    else:
        a_opt = a
        pos0, cell0, _ = build_supercell(lattice_factory(a_opt), nx, ny, nz)
        pos0, E0, _, _, _ = relax_positions_fixed_cell(
            pos0, cell0, potential,
            use_nlist=use_nlist, cutoff=cutoff,
            verbose=verbose, **(relax_params or {})
        )

    sigma, _, _, _ = stress_via_energy_fd_0K(
        pos0, cell0, potential,
        strain_eps=strain_eps, symmetric=symmetric,
        relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
        method=method, verbose=verbose
    )
    return {'sigma': sigma, 'E0': E0, 'pos0': pos0, 'cell0': cell0,'a_opt': a_opt}

# ...existing code...

# ------------------------
# 单分量应变-应力（中心差分，0K 固定晶胞）
# ------------------------
def stress_component_fd_0K(pos0_rel, cell, potential, i, j, strain_eps,
                           symmetric=True, relax_params=None,
                           use_nlist=False, cutoff=None,
                           volume_ref='reference', verbose=False):
    """
    计算某一分量 σ_ij(ε)，采用能量对应变的中心差分，0K 固定晶胞下每点重松弛原子。
    参数:
      pos0_rel, cell: 未应变基态构型（已在 cell 下松弛）
      i, j: 分量索引 0..2，对应 x,y,z
      strain_eps: 扰动幅度（标量）
      symmetric: True 时 i!=j 采用对称剪切 η_ij=η_ji=ε
      volume_ref: 'reference' 使用未应变体积，'current' 使用两次应变体积平均
    返回:
      sigma_ij: eV/Å^3
    说明:
      按你给定公式使用 σ_ij = -(1/V_ref) * ∂E/∂η_ji。为简洁按矩阵同位赋值，与上面的整矩阵函数一致。
    """
    if relax_params is None:
        relax_params = {}

    V0 = float(np.linalg.det(cell))

    # +eps
    eta_p = np.zeros((3,3))
    eta_p[i, j] = +strain_eps
    if symmetric and i != j:
        eta_p[j, i] = +strain_eps
    cell_p, pos_p0 = apply_strain(cell, pos0_rel, eta_p)
    pos_p, Ep, _, _, _ = relax_positions_fixed_cell(
        pos_p0, cell_p, potential,
        use_nlist=use_nlist, cutoff=cutoff,
        verbose=False, **relax_params
    )
    Vp = float(np.linalg.det(cell_p))

    # -eps
    eta_m = np.zeros((3,3))
    eta_m[i, j] = -strain_eps
    if symmetric and i != j:
        eta_m[j, i] = -strain_eps
    cell_m, pos_m0 = apply_strain(cell, pos0_rel, eta_m)
    pos_m, Em, _, _, _ = relax_positions_fixed_cell(
        pos_m0, cell_m, potential,
        use_nlist=use_nlist, cutoff=cutoff,
        verbose=False, **relax_params
    )
    Vm = float(np.linalg.det(cell_m))

    dE = (Ep - Em) / (2.0 * strain_eps)
    Vref = 0.5*(Vp+Vm) if volume_ref == 'current' else V0
    sigma_ij = - dE / Vref
    return sigma_ij

# ------------------------
# 扫描 σ_ij(ε) 并可保存与绘图
# ------------------------
def scan_sigma_component_vs_strain(pos0_rel, cell, potential, i, j, eps_list,
                                   symmetric=True, relax_params=None,
                                   use_nlist=False, cutoff=None,
                                   volume_ref='reference', verbose=False,
                                   save_csv_path=None):
    """
    对给定应变幅度列表 eps_list（标量列表）扫描 σ_ij(ε)。
    返回:
      eps_arr: (M,) 应变幅度
      sigma_eVA3: (M,) eV/Å^3
      sigma_GPa: (M,) GPa
    """
    eps_arr = np.asarray(eps_list, dtype=float)
    sig = []
    for eps in eps_arr:
        sij = stress_component_fd_0K(pos0_rel, cell, potential, i, j, eps,
                                     symmetric=symmetric, relax_params=relax_params,
                                     use_nlist=use_nlist, cutoff=cutoff,
                                     volume_ref=volume_ref, verbose=verbose)
        sig.append(sij)
    sigma_eVA3 = np.array(sig, dtype=float)
    sigma_GPa = sigma_eVA3 * 160.21766208  # 1 eV/Å^3 = 160.21766208 GPa

    if save_csv_path is not None:
        import os, csv
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        with open(save_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epsilon", f"sigma_{i}{j}_eV_A3", f"sigma_{i}{j}_GPa"])
            for e, s1, s2 in zip(eps_arr, sigma_eVA3, sigma_GPa):
                w.writerow([f"{e:.8e}", f"{s1:.8e}", f"{s2:.8e}"])

    return eps_arr, sigma_eVA3, sigma_GPa

def plot_sigma_vs_strain(eps_arr, sigma_vals, i, j, unit="GPa", title=None, save_png=None):
    """
    绘制 σ_ij vs ε。unit: 'GPa' 或 'eV/A^3'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图。请先 pip install matplotlib")
        return
    y = sigma_vals * 160.21766208 if unit.upper()=="GPA" else sigma_vals
    plt.figure()
    plt.plot(eps_arr, y, 'o-', lw=1.5, ms=4)
    plt.xlabel("strain ε")
    plt.ylabel(f"σ_{i}{j} ({unit})")
    plt.grid(True, ls="--", alpha=0.4)
    if title:
        plt.title(title)
    if save_png:
        plt.savefig(save_png, dpi=200, bbox_inches="tight")
    plt.show()

# ------------------------
# Example usage (do not run in module import if you don't want automatic run)
# ------------------------
if __name__ == "__main__":
    # Quick demo parameters (LJ placeholders)
    al_lat = make_fcc(a=4.05)  # 初始猜测
    nx,ny,nz = 2,2,2
    lj = LJPotential(eps=0.0103, sigma=2.5)
    out = strain_stress_0K_pipeline(
        lattice_factory=make_fcc, a=al_lat.a, nx=nx, ny=ny, nz=nz, potential=lj,
        strain_eps=1e-4, optimize_a=True, verbose=True,
        relax_params=dict(f_tol=1e-6, maxit=2000)
    )
    print("a_opt =", out['a_opt'])
    print("sigma (eV/Å^3) =\n", out['sigma'])

# 扫描示例：σ_xx 对不同应变幅度 ε 的关系（在 a_opt 基态上）
    try:
        eps_list = np.linspace(-5e-3, 5e-3, 11)  # 建议对称区间，便于观察线性区
        # 基态（已通过上面的 pipeline 获得）
        pos0 = out['pos0']
        cell0 = out['cell0']
        i, j = 0, 0  # σ_xx 与 ε_xx
        eps_arr, sigma_eVA3, sigma_GPa = scan_sigma_component_vs_strain(
            pos0, cell0, lj, i, j, eps_list,
            symmetric=True, relax_params=dict(f_tol=1e-6, maxit=2000),
            use_nlist=False, cutoff=None,
            volume_ref='reference', verbose=False,
            save_csv_path="./sigma_xx_scan.csv"
        )
        print("扫描完成，数据写入 ./sigma_xx_scan.csv")
        plot_sigma_vs_strain(eps_arr, sigma_eVA3, i, j, unit="GPa",
                             title="σ_xx vs ε_xx (0 K)", save_png="./sigma_xx_scan.png")
    except Exception as e:
        print("扫描/绘图示例出错：", e)
 