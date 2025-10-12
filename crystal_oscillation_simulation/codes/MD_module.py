"""
模块 2: 分子动力学(MD)模块

依赖（假定你已实现或导入）:
 - PotentialBase: 势函数基类(实现 energy_and_forces)
 - VerletNeighborList: 邻居表类(或 None 表示不使用)
 - virial_stress(positions, forces, cell): 返回势能部分 virial (eV/Å^3)
 - min_image(dr, cell, invcell): minimum image helper

单位:
 - length: Angstrom (Å)
 - time: femtosecond (fs)
 - energy: eV
 - mass: atomic mass unit (amu)
 - velocity: Å / fs
 - stress returned in eV/Å^3 ; multiply by 160.21766208 -> GPa
"""

import numpy as np
import time

# Physical constants / conversion
kB_eV_per_K = 8.617333262145e-5   # eV / K
amu_to_eVfs2_A2 = 1.036427e-2     # 1 amu * (Å/fs)^2 = 1.036427e-2 eV

# If your modules are in another file, import them:
# ------------------------
# Utilities
# ------------------------

def initialize_velocities(N, T, masses):
    """
    Initialize velocities from Maxwell-Boltzmann distribution at temperature T (K).
    masses: array of length N in amu
    returns velocities array shape (N,3) in Å/fs
    """
    # velocity variance per component: <v^2> = kB T / (m * conv) where conv converts m*(Å/fs)^2 -> eV
    masses = np.asarray(masses)
    sigma_v = np.sqrt(kB_eV_per_K * T / (masses * amu_to_eVfs2_A2))
    # draw from normal with 0 mean, sigma_v per particle for each component
    vel = np.random.normal(0.0, 1.0, size=(len(masses), 3)) * sigma_v[:, None]
    # remove center-of-mass momentum
    v_com = np.average(vel, axis=0, weights=masses)
    vel -= v_com
    return vel

def kinetic_energy(velocities, masses):
    """
    velocities (N,3) Å/fs, masses in amu
    returns kinetic energy in eV
    """
    m = masses[:, None]
    # KE = 0.5 * sum_i m_i v_i^2 * conv  (conv = amu*(Å/fs)^2 -> eV)
    ke = 0.5 * np.sum(m * velocities**2) * amu_to_eVfs2_A2
    return ke

def temperature_from_kinetic(velocities, masses, dof_correction=0):
    """
    returns instantaneous temperature (K)
    degrees of freedom = 3N - dof_correction (e.g., -3 for COM removed)
    """
    N = len(masses)
    ke = kinetic_energy(velocities, masses)
    dof = 3*N - dof_correction
    T = (2.0 * ke) / (dof * kB_eV_per_K)
    return T

# ------------------------
# Virial stress (full: kinetic + potential)
# ------------------------
def virial_stress_full(positions, velocities, forces, cell, masses):
    """
    Return Cauchy stress tensor sigma (3x3) in eV/Å^3
    sigma = (1/V) [ sum_i m_i v_i ⊗ v_i + 1/2 sum_{i≠j} r_ij ⊗ F_ij ]
    Here: use potential virial W_pot = 0.5 * sum_i (r_i ⊗ F_i) as implemented elsewhere
    positions: (N,3)
    velocities: (N,3)
    forces: (N,3) (pairwise-consistent)
    masses: array (N) in amu
    """
    vol = np.linalg.det(cell)
    # kinetic term (units mass*vel^2 -> convert to eV)
    # m (amu) * v(Å/fs)^2 * amu_to_eVfs2_A2 -> eV
    K = np.zeros((3,3))
    # form sum m v_i a v_i b
    for i in range(len(masses)):
        m_e = masses[i] * amu_to_eVfs2_A2
        K += m_e * np.outer(velocities[i], velocities[i])
    K = K / vol  # eV/Å^3
    # potential virial (use potential-specific calculation or generic half r_i ⊗ F_i)
    # We'll compute potential virial as 0.5 * sum_i r_i ⊗ F_i (consistent with previous module)
    W = np.zeros((3,3))
    for i in range(len(positions)):
        W += np.outer(positions[i], forces[i])
    W = 0.5 * W / vol
    sigma = K + W
    return sigma

# ------------------------
# Integrator: Velocity-Verlet with thermostat hooks
# ------------------------
def velocity_verlet_step(positions, velocities, forces, cell, dt, masses,
                         potential, neighbor_list=None):
    """
    One velocity-Verlet step without thermostat.
    masses in amu, velocities in Å/fs, forces in eV/Å
    returns updated (positions, velocities, forces, energy)
    """
    N = len(positions)
    invcell = np.linalg.inv(cell)
    # 1) update positions
    # accelerations a = F / m  (units: (eV/Å) / amu -> (Å/fs^2) after conversion)
    # F (eV/Å) -> a (Å/fs^2) = F / (m * conv) with conv = amu_to_eVfs2_A2
    acc = forces / (masses[:, None] * amu_to_eVfs2_A2)
    positions += velocities * dt + 0.5 * acc * dt * dt
    # wrap positions into cell
    fracs = (invcell.dot(positions.T)).T
    fracs -= np.floor(fracs)
    positions = (cell.dot(fracs.T)).T
    # 2) compute new forces
    energy, new_forces = potential.energy_and_forces(positions, cell, neighbor_list)
    # 3) update velocities with new acceleration
    new_acc = new_forces / (masses[:, None] * amu_to_eVfs2_A2)
    velocities += 0.5 * (acc + new_acc) * dt
    return positions, velocities, new_forces, energy

# ------------------------
# Langevin thermostat (Ornstein-Uhlenbeck) - applied after VV step or in BAOAB splitting
# ------------------------
def apply_langevin(velocities, masses, gamma, dt, T):
    """
    velocities: (N,3) Å/fs
    gamma: friction coefficient in 1/fs
    dt: time step fs
    masses: array amu
    T: temperature K
    Performs simple velocity update: v <- e^{-gamma dt} v + sqrt((1-e^{-2gamma dt}) * kT/m) * R
    """
    N = len(masses)
    # exponential factor
    expf = np.exp(-gamma * dt)
    # random strength factor per particle
    sigma = np.sqrt((1.0 - expf**2) * (kB_eV_per_K * T) / (masses * amu_to_eVfs2_A2))
    # broadcast to (N,3)
    rand = np.random.normal(0.0, 1.0, size=(N,3))
    velocities *= expf
    velocities += rand * sigma[:, None]
    # remove net momentum drift (optional)
    return velocities

# ------------------------
# Simple Nosé-Hoover chain (single thermostat variable)
# ------------------------
class NoseHoover:
    def __init__(self, Q, dt):
        """
        Q: thermostat mass (in eV * fs^2) — choose roughly Q ~ N_dof * kB * T * tau^2,
           where tau is relaxation time in fs.
        dt: time step fs
        We'll use a simple symmetric Trotter integration of the thermostat variable.
        """
        self.xi = 0.0   # thermostat velocity
        self.eta = 0.0  # thermostat position (not used except monitoring)
        self.Q = Q
        self.dt = dt

    def thermostat_step(self, velocities, masses, T, dof):
        """
        Apply single Nose-Hoover thermostat half-step: scale velocities
        Using simple algorithm:
          v <- v * exp(-xi*dt/2)
        And update xi using kinetic energy deviation
        """
        # half step scaling
        factor = np.exp(-0.5 * self.xi * self.dt)
        velocities *= factor
        # compute current kinetic energy
        ke = kinetic_energy(velocities, masses)  # eV
        # xi full-step update using equation of motion: xi_dot = (2K - dof*kT) / Q
        xi_dot = (2.0 * ke - dof * kB_eV_per_K * T) / self.Q
        self.xi += xi_dot * self.dt
        # another half-step scaling
        velocities *= factor
        # update eta (optional)
        self.eta += self.xi * self.dt
        return velocities

# ------------------------
# High-level MD runner
# ------------------------
def run_md(positions, cell, masses, potential,
           n_steps, dt,
           temperature=None,
           thermostat='langevin', thermostat_params=None,
           neighbor_list=None, nlist_update_interval=10,
           sample_interval=10, equil_steps=0,
           save_xyz=None, verbose=True):
    """
    High-level MD driver.
    - positions: (N,3) initial positions (Å)
    - cell: 3x3 matrix
    - masses: array of length N in amu
    - potential: PotentialBase instance
    - n_steps: total MD steps
    - dt: time step fs
    - temperature: if None -> NVE (no thermostat), else NVT using chosen thermostat
    - thermostat: 'langevin' or 'nose-hoover'
    - thermostat_params: dict for thermostat (gamma for langevin in 1/fs, Q and tau for NH)
    - neighbor_list: VerletNeighborList instance or None
    - nlist_update_interval: rebuild neighbor list every this many steps if provided
    - sample_interval: record stats every this many steps
    - equil_steps: number of steps before starting to sample averages
    - save_xyz: filename to save trajectory in XYZ appended each sample (optional)
    Returns: dictionary with time series: energy, temp, stress (list), positions last state
    """
    N = len(positions)
    masses = np.asarray(masses)
    invcell = np.linalg.inv(cell)
    # initialize neighbor list if provided and not yet built
    if neighbor_list is not None and neighbor_list.list is None:
        neighbor_list.build(positions, cell)
    # initial forces and energy
    energy, forces = potential.energy_and_forces(positions, cell, neighbor_list)
    # initial velocities
    if temperature is not None:
        velocities = initialize_velocities(N, temperature, masses)
    else:
        velocities = np.zeros((N,3))
    # initialize thermostat
    nh = None
    if thermostat == 'nose-hoover' and temperature is not None:
        # choose Q: if not provided, pick Q ~ dof*kT*tau^2
        dof = 3*N - 3
        tau = thermostat_params.get('tau', 100.0) if thermostat_params else 100.0
        Q = thermostat_params.get('Q', dof * kB_eV_per_K * temperature * tau*tau)
        nh = NoseHoover(Q=Q, dt=dt)
    # data containers
    times = []
    temps = []
    energies = []
    stresses = []  # store full 3x3 matrices as flattened arrays
    ke_list = []
    # trajectory file
    traj_file = None
    if save_xyz is not None:
        traj_file = open(save_xyz, 'w')

    # warm-up: run integrator loop
    t0 = time.time()
    samples = 0
    for step in range(1, n_steps+1):
        # optional neighbor list rebuild
        if neighbor_list is not None and (step % nlist_update_interval == 0):
            neighbor_list.build(positions, cell)
        # Integrator: velocityVerlet step
        positions, velocities, forces, potential_energy = velocity_verlet_step(
            positions, velocities, forces, cell, dt, masses, potential, neighbor_list
        )
        # thermostat
        if temperature is not None:
            if thermostat == 'langevin':
                gamma = thermostat_params.get('gamma', 0.02) if thermostat_params else 0.02
                velocities = apply_langevin(velocities, masses, gamma, dt, temperature)
            elif thermostat == 'nose-hoover':
                dof = 3*N - 3
                velocities = nh.thermostat_step(velocities, masses, temperature, dof)
            # else: unsupported, fallback to NVE
        # sampling
        if step > equil_steps and (step % sample_interval == 0):
            samples += 1
            ke = kinetic_energy(velocities, masses)
            T_inst = temperature_from_kinetic(velocities, masses, dof_correction=3)
            sigma = virial_stress_full(positions, velocities, forces, cell, masses)
            times.append(step*dt)
            temps.append(T_inst)
            energies.append(potential_energy + ke)
            ke_list.append(ke)
            stresses.append(sigma.copy())
            # write xyz snapshot
            if traj_file is not None:
                traj_file.write(f"{len(positions)}\n")
                traj_file.write(f"Step {step} Time {step*dt:.4f} fs Epot {potential_energy:.6f} eV\n")
                for i in range(len(positions)):
                    x,y,z = positions[i]
                    traj_file.write(f"X {x:.8f} {y:.8f} {z:.8f}\n")
        # periodic progress print
        if verbose and (step % max(1, n_steps//10) == 0):
            print(f"[MD] step {step}/{n_steps}  t={step*dt:.1f} fs")
    t1 = time.time()
    if traj_file is not None:
        traj_file.close()

    # compile results
    results = {
        'times_fs': np.array(times),
        'temperatures_K': np.array(temps),
        'total_energies_eV': np.array(energies),
        'kinetic_energies_eV': np.array(ke_list),
        'stresses_eV_per_A3': np.array(stresses),  # shape (nsamples, 3, 3)
        'positions_final': positions,
        'velocities_final': velocities,
        'potential_energy_final': potential_energy,
        'walltime_s': t1 - t0,
        'samples': samples
    }
    return results

# ------------------------
# Example helper for running short NVT (do not run on import)
# ------------------------
if __name__ == "__main__":
    # Usage example (requires potential and build_supercell in your env)
    from opt_method import make_fcc, build_supercell, VerletNeighborList
    from potential import LJPotential
    a0 = 4.05
    lat = make_fcc(a0)
    pos, cell, syms = build_supercell(lat, 2,2,2)
    N = len(pos)
    masses = np.array([26.9815385] * N)  # Al mass in amu
    lj = LJPotential(eps=0.0103, sigma=2.5)
    nlist = VerletNeighborList(cutoff=2.5*2.5, skin=0.3)  # adjust cutoff as needed
    nlist.build(pos, cell)
    res = run_md(pos, cell, masses, lj,
                 n_steps=2000, dt=1.0,
                 temperature=300.0,
                 thermostat='langevin', thermostat_params={'gamma':0.05},
                 neighbor_list=nlist, nlist_update_interval=20,
                 sample_interval=10, equil_steps=200,
                 save_xyz=None, verbose=True)
    print("MD done. samples:", res['samples'], "walltime(s):", res['walltime_s'])
