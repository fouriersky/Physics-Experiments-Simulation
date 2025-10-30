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

def _resolve_min_method(potential, method):
    """
    返回实际使用的方法：'external' 或 'cg'
    """
    if method in ("external", "cg"):
        return method
    # auto 选择：若 potential 提供 relax_positions_fixed_cell，则用 external
    return "external" if hasattr(potential, "relax_positions_fixed_cell") else "cg"

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
                            volume_ref='reference', verbose=False, method='auto'):
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
    method_eff = _resolve_min_method(potential, method)
    # 1) 基态松弛（未应变）
    pos0_rel, E0, _, _, _ = relax_positions_fixed_cell(
        positions, cell, potential,
        use_nlist=use_nlist, cutoff=cutoff,
        verbose=verbose, method=method_eff, **relax_params
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
                verbose=False, method=method_eff, **relax_params
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

# NPT optimziation lattice parameter a
def optimize_scalar_a_T_NPT(lattice_factory, a_init, nx, ny, nz, potential,
                            T_K=300.0, P_bar=0.0,
                            dt_fs=1.0, total_steps=60000, equil_steps=20000,
                            block_steps=200, seed=12345,
                            tau_t_ps=0.1, tau_p_ps=1.0,
                            keep_tmp_files=False, verbose=True):
    """
    有限温度下的未微扰晶格常数优化（NPT，<P>=P_bar，默认 P=0）。
    流程：
      1) 以 a_init 构建正交超胞；
      2) 运行 LAMMPS NPT (temp T_K, iso P_bar)，产出阶段时间平均 lx, ly, lz；
      3) 返回 a_T = 平均后等效晶格常数（对 cubic: 三方向取平均）。
    注意：
      - 假设使用正交对角 cell 的构造（build_supercell），适用于 fcc/diamond 的常规胞表达。
      - 若体系非各向同性，请自行使用三个方向的平均长度分别除以 nx,ny,nz。
    返回:
      dict: {
        'a_T': float,
        'L_avg': (Lx, Ly, Lz),
        'cell_avg': 3x3 diag(Lx, Ly, Lz),
        'T_K': T_K,
        'P_bar': P_bar
      }
    """
    import os, shutil, tempfile, subprocess
    import numpy as np
    from potential import DirectLAMMPSEAMPotential  # 仅复用其写 data 的 triclinic 工具

    # 1) 构建初始超胞（正交）
    positions, cell, _ = build_supercell(lattice_factory(a_init), nx, ny, nz)
    mass_amu = float(getattr(potential, "mass", 12.0))

    # 2) 解析势文件与 pair_style/pair_coeff
    if hasattr(potential, "lcbop_file"):
        pot_path = potential.lcbop_file
        pair_style = getattr(potential, "pair_style", "lcbop")
        element = getattr(potential, "element", "C")
        pair_lines = [f"pair_style {pair_style}",
                      f"pair_coeff * * {{POT}} {element}"]
    elif hasattr(potential, "eam_file"):
        pot_path = potential.eam_file
        pair_style = getattr(potential, "pair_style", "eam/alloy")
        element = getattr(potential, "element", "Al")
        if "alloy" in pair_style.lower():
            pair_lines = [f"pair_style {pair_style}",
                          f"pair_coeff * * {{POT}} {element}"]
        else:
            pair_lines = [f"pair_style {pair_style}",
                          f"pair_coeff * * {{POT}}"]
    else:
        raise RuntimeError("unknown potential backend: 需提供 lcbop_file 或 eam_file")

    lmp_cmd = getattr(potential, "lmp_cmd", "lmp.exe")

    # 3) 写入 LAMMPS data 与输入脚本
    tmp_dir = tempfile.mkdtemp(prefix="npt_aopt_")
    try:
        pot_base = os.path.basename(pot_path)
        shutil.copy2(pot_path, os.path.join(tmp_dir, pot_base))
        data_path = os.path.join(tmp_dir, "data.in")
        avg_path  = os.path.join(tmp_dir, "avg_box.txt")
        dump_scaled_path = os.path.join(tmp_dir, "dump_scaled.lammpstrj")
        in_path   = os.path.join(tmp_dir, "in.npt")

        # 写 data（任意晶胞均可；当前 cell 为正交）
        # 返回的 Q 未使用：NPT 只关心盒长的平均
        DirectLAMMPSEAMPotential._write_data_atomic_triclinic(
            data_path, cell, positions, mass_amu=mass_amu
        )

        T = float(T_K)
        P = float(P_bar)
        dt_ps = float(dt_fs) / 1000.0
        Tdamp = float(tau_t_ps)
        Pdamp = float(tau_p_ps)
        n_eq  = int(equil_steps)
        n_prd = int(total_steps - equil_steps)
        n_prd = max(n_prd, 1)

        # 每 block_steps 步累计一次平均
        bs = int(block_steps)
        if bs <= 0:
            bs = 100

        # 组装输入：先热化 run n_eq，再产出阶段 run n_prd 并用 fix ave/time 写 lx,ly,lz
        lines = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f'read_data "{data_path}"',
            "",
            "### interactions",
            pair_lines[0],
            pair_lines[1].replace("{POT}", pot_base),
            f"mass 1 {mass_amu:.6f}",
            "",
            "neighbor 2.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            "",
            f"variable T equal {T:.16g}",
            f"variable P equal {P:.16g}",
            f"variable SEED equal {int(seed)}",
            f"timestep {dt_ps:.16g}",
            "",
            "velocity all create ${T} ${SEED} mom yes rot yes dist gaussian",
            # NPT 等温等压（各向同性），pressure 单位：bar（units metal）
            f"fix nptfix all npt temp ${{T}} ${{T}} {Tdamp:.6g} iso ${{P}} ${{P}} {Pdamp:.6g}",
            "thermo_style custom step temp press pe ke lx ly lz vol",
            f"thermo {bs}",
            "",
            f"run {n_eq}",
            "",
            "# production with box averaging",
            "unfix nptfix",
            f"fix nptfix all npt temp ${{T}} ${{T}} {Tdamp:.6g} iso ${{P}} ${{P}} {Pdamp:.6g}",
            "variable Lx equal lx",
            "variable Ly equal ly",
            "variable Lz equal lz",
            f'fix avgbox all ave/time 1 {bs} {bs} v_Lx v_Ly v_Lz file "{avg_path}"',
            # 记录分数坐标 xs,ys,zs（方便映射到平均盒）
            f'dump dsc all custom 1 "{dump_scaled_path}" id type xs ys zs',
            "dump_modify dsc sort id",
            f"run {n_prd}",
            "unfix avgbox",
            "undump dsc",
            "unfix nptfix",
            ""
        ]
        with open(in_path, "w", newline="\n") as f:
            f.write("\n".join(lines))

        # 4) 调 LAMMPS
        proc = subprocess.run([lmp_cmd, "-in", in_path],
                              cwd=tmp_dir, capture_output=True, text=True, timeout=max(60, total_steps//50))
        if proc.returncode != 0:
            raise RuntimeError(
                f"NPT 失败，返回码 {proc.returncode}\n--- STDOUT ---\n{(proc.stdout or '')[-1500:]}\n"
                f"--- STDERR ---\n{(proc.stderr or '')[-1500:]}"
            )

        # 5) 解析 avg_box.txt
        Lx, Ly, Lz = [], [], []
        with open(avg_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Step"):
                    continue
                parts = line.split()
                # fix ave/time 输出通常为：step lx ly lz
                if len(parts) >= 4:
                    try:
                        lx = float(parts[1]); ly = float(parts[2]); lz = float(parts[3])
                        Lx.append(lx); Ly.append(ly); Lz.append(lz)
                    except Exception:
                        pass
        if not Lx:
            raise RuntimeError("未能从 avg_box.txt 解析到盒长数据。")

        Lx_avg = float(np.mean(Lx))
        Ly_avg = float(np.mean(Ly))
        Lz_avg = float(np.mean(Lz))

        # 6) 等效晶格常数（适用于 cubic）
        a_T = (Lx_avg / nx + Ly_avg / ny + Lz_avg / nz) / 3.0
        cell_avg = np.diag([Lx_avg, Ly_avg, Lz_avg])
        # 解析 dump_scaled 的最后一帧分数坐标 xs,ys,zs，并映射到平均盒
        N = len(positions)
        def _read_last_scaled_coords(path, N):
            # 读取 lammpstrj，定位最后一个 "ITEM: ATOMS" 块
            with open(path, "r") as f:
                lines = f.readlines()
            start_idx = None
            for idx in range(len(lines)-1, -1, -1):
                if lines[idx].startswith("ITEM: ATOMS"):
                    start_idx = idx + 1
                    break
            if start_idx is None:
                raise RuntimeError("未在 dump_scaled 中找到 ATOMS 块")
            frac = np.zeros((N, 3), dtype=float)
            for k in range(N):
                parts = lines[start_idx + k].split()
                # 格式: id type xs ys zs
                i = int(parts[0]) - 1
                xs = float(parts[2]); ys = float(parts[3]); zs = float(parts[4])
                frac[i] = (xs, ys, zs)
            return frac

        frac_last = _read_last_scaled_coords(dump_scaled_path, N)
        pos_avg = (cell_avg @ frac_last.T).T  # 映射到平均盒下的笛卡尔坐标
        
        if verbose:
            print(f"[NPT a-opt] <Lx,Ly,Lz>=({Lx_avg:.6f}, {Ly_avg:.6f}, {Lz_avg:.6f}) Å -> a_T={a_T:.6f} Å @ {T_K} K")

        return {
            'a_T': a_T,
            'L_avg': (Lx_avg, Ly_avg, Lz_avg),
            'cell_avg': cell_avg,
            'pos_avg': pos_avg,
            'T_K': T_K,
            'P_bar': P_bar
        }

    finally:
        if not keep_tmp_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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

def _tensor_to_voigt_sigma(s):
    """σ(3x3)->Voigt: [sxx, syy, szz, syz, sxz, sxy]"""
    s = np.asarray(s, dtype=float)
    return np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]], dtype=float)

def _eta_from_voigt(j, eps, convention="engineering"):
    """
    根据 Voigt 分量 j 和标量 eps 生成对称应变张量 η。
    convention='engineering' 时，eps 对剪切为工程剪切 γ，内部用 εij=γ/2。
    """
    E = np.zeros((3,3), dtype=float)
    if j == 0:
        E[0,0] = eps
    elif j == 1:
        E[1,1] = eps
    elif j == 2:
        E[2,2] = eps
    elif j == 3:
        val = 0.5*eps if convention == "engineering" else eps
        E[1,2] = E[2,1] = val
    elif j == 4:
        val = 0.5*eps if convention == "engineering" else eps
        E[0,2] = E[2,0] = val
    elif j == 5:
        val = 0.5*eps if convention == "engineering" else eps
        E[0,1] = E[1,0] = val
    else:
        raise ValueError("Voigt 索引应为 0..5")
    return E

def sigma_vs_single_voigt_strain_0K(pos0, cell0, potential, j, eps_list,
                                    delta_fd=2e-4, convention="engineering",
                                    relax_params=None, use_nlist=False, cutoff=None,
                                    volume_ref='reference', verbose=False, method='auto'):
    """
    对于给定的 Voigt 分量 j，扫描若干 eps，返回每个 eps 下的整张 σ(3x3)。
    实现：对每个 eps，先将 η0(j,eps) 作用到 (pos0,cell0) 得到基态胞，再调
    stress_via_energy_fd_0K 在该基态上计算 σ。
    返回：(eps_arr, [σ_tensor_list])
    """
    if relax_params is None:
        relax_params = {}
    eps_arr = np.asarray(eps_list, dtype=float)
    sigmas = []
    for e in eps_arr:
        eta0 = _eta_from_voigt(j, e, convention=convention)
        cell_b, pos_b = apply_strain(cell0, pos0, eta0)
        sigma_b, _, _, _ = stress_via_energy_fd_0K(
            pos_b, cell_b, potential,
            strain_eps=delta_fd, symmetric=True,
            relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
            volume_ref=volume_ref, verbose=verbose, method=method
        )
        sigmas.append(sigma_b)
    return eps_arr, sigmas

def build_C_0K_central_difference(pos0, cell0, potential,
                                  strain_eps=1e-3,       # 标量；对剪切为工程剪切 γ
                                  delta_fd=2e-4,         # stress_via_energy_fd_0K 内部用的微小差分
                                  convention="engineering",
                                  relax_params=None, use_nlist=False, cutoff=None,
                                  volume_ref='reference', verbose=False, method='auto',
                                  to_GPa=True, symmetrize=True):
    """
    0K 下用工程剪切记号做中心差分，直接构建 C 的 6 列：
      C[:,j] = (σ(+ε_j) - σ(-ε_j)) / (2 ε_j)
    其中 ε_j 对 j=3..5 为工程剪切 γ；施加到应变张量时用 εij=γ/2。
    """
    if relax_params is None:
        relax_params = {}
    eps = float(strain_eps)
    C = np.zeros((6, 6), dtype=float)

    for j in range(6):
        # +eps（工程剪切/拉伸）
        eta_p = _eta_from_voigt(j, +eps, convention=convention)
        cell_p, pos_p = apply_strain(cell0, pos0, eta_p)
        sigma_p, _, _, _ = stress_via_energy_fd_0K(
            pos_p, cell_p, potential,
            strain_eps=delta_fd, symmetric=True,
            relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
            volume_ref=volume_ref, verbose=verbose, method=method
        )

        # -eps
        eta_m = _eta_from_voigt(j, -eps, convention=convention)
        cell_m, pos_m = apply_strain(cell0, pos0, eta_m)
        sigma_m, _, _, _ = stress_via_energy_fd_0K(
            pos_m, cell_m, potential,
            strain_eps=delta_fd, symmetric=True,
            relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
            volume_ref=volume_ref, verbose=verbose, method=method
        )

        # 列向量（Voigt）：σ_vec = [sxx, syy, szz, syz, sxz, sxy]
        s_p = _tensor_to_voigt_sigma(sigma_p)
        s_m = _tensor_to_voigt_sigma(sigma_m)
        C[:, j] = (s_p - s_m) / (2.0 * eps)

    if symmetrize:
        C = 0.5 * (C + C.T)
    if to_GPa:
        C = C * 160.21766208
    return C

def build_C_0K_via_scans(pos0, cell0, potential,
                         eps_max=2e-3, n_points=5, delta_fd=2e-4,
                         convention="engineering", to_GPa=True,
                         relax_params=None, use_nlist=False, cutoff=None,
                         volume_ref='reference', verbose=False, method='cg',
                         symmetrize=True):
    """
    0K 下通过多点线性拟合得到 C（默认工程剪切记号）。
    对每个列 j（Voigt 0..5），扫 eps∈[-eps_max, ..., +eps_max]（包含0），
    在每个 eps 的基态上计算整张 σ，然后对 σ_vec vs eps 做带截距线性拟合，取斜率为列。
    """
    if relax_params is None:
        relax_params = {}
    # 对称 eps 列表（含 0）
    if n_points < 3:
        n_points = 3
    half = (n_points - 1) // 2
    eps_list = np.linspace(-eps_max, eps_max, 2*half+1)

    C = np.zeros((6,6), dtype=float)  # eV/Å^3 by default
    voigt_pairs = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]

    for j in range(6):
        e_arr, sigma_tensors = sigma_vs_single_voigt_strain_0K(
            pos0, cell0, potential, j, eps_list,
            delta_fd=delta_fd, convention=convention,
            relax_params=relax_params, use_nlist=use_nlist, cutoff=cutoff,
            volume_ref=volume_ref, verbose=verbose, method=method
        )
        # 组装每个 eps 下的 Voigt σ 向量
        S = np.vstack([_tensor_to_voigt_sigma(Sij) for Sij in sigma_tensors])  # (K,6)
        # 对每个分量 i 拟合 σ_i = a_i*eps + b_i，取 a_i 为 C_{i,j}
        for i in range(6):
            a_i, b_i = np.polyfit(e_arr, S[:, i], 1)
            C[i, j] = a_i

    # 对工程剪切：已用 γ 作为 eps，列无需再缩放；若改为 tensor 记号，请将 C[:,3:6] *= 0.5
    if convention == "engineering":
        C[:, 3:6] *= 0.5
    if symmetrize:
        C = 0.5*(C + C.T)
    if to_GPa:
        C = C * 160.21766208  # eV/Å^3 -> GPa
    return C

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
        relax_params=dict(f_tol=1e-6, maxit=2000),  
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
    
