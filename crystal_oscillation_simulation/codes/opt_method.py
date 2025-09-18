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
    frac = invcell.dot(dr)
    # wrap to [-0.5, 0.5)
    frac -= np.round(frac)
    return cell.dot(frac)

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
        self.rc = rc if rc is not None else 2.5*sigma
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


# ------------------------
# Virial stress (potential part) for 0K
# ------------------------
def virial_stress(positions, forces, cell):
    """
    potential part only; in units eV/Å^3
    sigma = (1/2V) * sum_i ( r_i ⊗ F_i )
    Note: requires forces computed consistently (pairwise with min-image)
    """
    vol = np.linalg.det(cell)
    W = np.zeros((3,3))
    for i in range(len(positions)):
        W += np.outer(positions[i], forces[i])
    sigma = 0.5 * W / vol
    return sigma

# ------------------------
# FIRE optimizer (for 0K ions position optimization)
# ------------------------
def fire_minimize(positions, cell, potential, neighbor_list=None,
                  f_tol=1e-6, maxit=5000, dt_init=0.002, verbose=False):
    """
    Simplified FIRE algorithm:
    positions: (N,3) array (cartesian)
    cell: 3x3
    potential: PotentialBase
    neighbor_list: VerletNeighborList or None
    returns: positions_relaxed, energy, forces, iterations
    """
    positions = positions.copy()
    N = len(positions)
    velocities = np.zeros_like(positions)
    dt = dt_init
    alpha = 0.1
    N_positive = 0
    # initial forces
    energy, forces = potential.energy_and_forces(positions, cell, neighbor_list)
    for it in range(maxit):
        fmax = np.max(np.linalg.norm(forces, axis=1))
        if fmax < f_tol:
            if verbose:
                print(f"[FIRE] converged at it={it}, fmax={fmax:.3e}")
            return positions, energy, forces, it
        # half step velocity update
        velocities += 0.5 * forces * dt
        # compute power P = sum F·v
        P = np.sum(forces * velocities)
        vnorm = np.linalg.norm(velocities)
        fnorm = np.linalg.norm(forces)
        if vnorm > 0 and fnorm > 0:
            velocities = (1 - alpha) * velocities + alpha * (forces / fnorm) * vnorm
        # integrate positions
        positions += velocities * dt
        # enforce PBC (wrap into [0,1))
        invcell = np.linalg.inv(cell)
        fracs = (invcell.dot(positions.T)).T
        fracs -= np.floor(fracs)
        positions = (cell.dot(fracs.T)).T
        # recompute forces
        energy, forces = potential.energy_and_forces(positions, cell, neighbor_list)
        # finish velocity step
        velocities += 0.5 * forces * dt
        # FIRE parameter updates
        if P > 0:
            N_positive += 1
            if N_positive > 5:
                dt = min(dt * 1.1, 0.05)
                alpha *= 0.99
        else:
            N_positive = 0
            dt *= 0.5
            velocities[:] = 0.0
            alpha = 0.1
    # not converged
    if verbose:
        print(f"[FIRE] not converged after maxit={maxit}, fmax={np.max(np.linalg.norm(forces,axis=1)):.3e}")
    return positions, energy, forces, maxit

# ------------------------
# Relax positions at fixed cell (wrapper)
# ------------------------
def relax_positions_fixed_cell(positions, cell, potential,
                               use_nlist=False, cutoff=None, skin=0.3,
                               f_tol=1e-6, maxit=5000, verbose=False):
    """
    Relax internal coordinates at fixed cell.
    If use_nlist True, build Verlet list with cutoff (required).
    Returns: positions_relaxed, energy, forces, iter_count, neighbor_list_used
    """
    neighbor_list = None
    if use_nlist:
        if cutoff is None:
            raise ValueError("cutoff must be provided when use_nlist=True")
        neighbor_list = VerletNeighborList(cutoff=cutoff, skin=skin)
        neighbor_list.build(positions, cell)
    pos_relaxed, E, F, it = fire_minimize(positions, cell, potential,
                                          neighbor_list=neighbor_list,
                                          f_tol=f_tol, maxit=maxit, verbose=verbose)
    return pos_relaxed, E, F, it, neighbor_list

# ------------------------
# Simple 1D search for optimal lattice constant a
# (coarse-to-fine grid search)
# ------------------------
def relax_lattice_scalar_search(lattice_factory, a0, nx, ny, nz,
                                potential, a_rel_range=(0.98, 1.02),
                                n_coarse=9, n_fine=11,
                                relax_params=None, use_nlist=False, cutoff=None):
    """
    lattice_factory: function like make_fcc that takes a and returns Lattice
    a0: initial lattice constant (Å)
    nx,ny,nz: supercell size
    potential: PotentialBase instance
    a_rel_range: tuple (min_factor, max_factor) relative to a0
    n_coarse: number of samples in coarse scan
    n_fine: number of samples in fine scan
    relax_params: dict forwarded to relax_positions_fixed_cell (f_tol, maxit, etc.)
    Returns: dict with best_a, best_positions, best_cell, best_energy, all_data (list)
    """
    if relax_params is None:
        relax_params = {}
    a_min = a0 * a_rel_range[0]
    a_max = a0 * a_rel_range[1]
    # coarse scan
    a_list = np.linspace(a_min, a_max, n_coarse)
    coarse_results = []
    for a in a_list:
        lattice = lattice_factory(a)
        pos, cell, syms = build_supercell(lattice, nx, ny, nz)
        pos_relaxed, E, F, it, nlist = relax_positions_fixed_cell(pos, cell, potential,
                                                                  use_nlist=use_nlist,
                                                                  cutoff=cutoff,
                                                                  **relax_params)
        E_per_atom = E / len(pos_relaxed)
        coarse_results.append((a, E_per_atom, pos_relaxed, cell))
    # find min in coarse
    a_best_coarse, _, _, _ = min(coarse_results, key=lambda x: x[1])
    # fine scan around best coarse
    fine_min = max(a_min, a_best_coarse - (a_max-a_min)/n_coarse)
    fine_max = min(a_max, a_best_coarse + (a_max-a_min)/n_coarse)
    a_list_fine = np.linspace(fine_min, fine_max, n_fine)
    fine_results = []
    for a in a_list_fine:
        lattice = lattice_factory(a)
        pos, cell, syms = build_supercell(lattice, nx, ny, nz)
        pos_relaxed, E, F, it, nlist = relax_positions_fixed_cell(pos, cell, potential,
                                                                  use_nlist=use_nlist,
                                                                  cutoff=cutoff,
                                                                  **relax_params)
        E_per_atom = E / len(pos_relaxed)
        fine_results.append((a, E_per_atom, pos_relaxed, cell))
    a_best, Ebest, pos_best, cell_best = min(fine_results, key=lambda x: x[1])
    return {
        'best_a': a_best,
        'best_energy_per_atom': Ebest,
        'best_positions': pos_best,
        'best_cell': cell_best,
        'coarse': coarse_results,
        'fine': fine_results
    }

# ------------------------
# Example usage (do not run in module import if you don't want automatic run)
# ------------------------
if __name__ == "__main__":
    # Quick demo parameters (LJ placeholders)
    al_lat = make_fcc(a=4.05)
    nx,ny,nz = 2,2,2  # small supercell for test
    pos, cell, syms = build_supercell(al_lat, nx, ny, nz)
    print("Test system: N atoms =", len(pos), "cell vol (Å^3) =", np.linalg.det(cell))
    # LJ placeholder params (NOT physical for Al/C)
    lj = LJPotential(eps=0.0103, sigma=2.5)
    # Relax positions at fixed cell
    pos_rel, E, F, it, nlist = relax_positions_fixed_cell(pos, cell, lj,
                                                          use_nlist=False,
                                                          f_tol=1e-5, maxit=2000, verbose=True)
    print("Relax done: E_total (eV) =", E, "max |F| (eV/Å) =", np.max(np.linalg.norm(F,axis=1)))
    # Virial stress (0K)
    sigma = virial_stress(pos_rel, F, cell)
    print("Virial stress (eV/Å^3):\n", sigma)
    print("Virial stress (GPa):\n", sigma * 160.21766208)
    # Optionally do lattice constant search (slow for large cells)
    # res = relax_lattice_scalar_search(make_fcc, a0=4.05, nx=2,ny=2,nz=2, potential=lj,
    #                                   a_rel_range=(0.98,1.02), n_coarse=7, n_fine=9,
    #                                   relax_params={'f_tol':1e-5,'maxit':1000}, use_nlist=False)
    # print("Best a:", res['best_a'], "E/atom:", res['best_energy_per_atom'])
 