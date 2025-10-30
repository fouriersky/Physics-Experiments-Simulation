
"""
calculate C-diamond stiffness tensor at T(K) = 100 200 300 400 500 600 700 800 
                                        P = 1bar
"""

from potential import DirectLAMMPSLCBOPPotential
from opt_method import optimize_scalar_a_T_NPT,make_diamond,strain_stress_0K_pipeline, build_C_0K_central_difference 
from MD_module import build_C_by_central_difference,build_C_by_central_difference_NPT

def main(T):
    nx, ny, nz = 3, 3, 3
    a_guess = 3.6
    lenT = len(T)
    pot = DirectLAMMPSLCBOPPotential(
        lcbop_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
        lmp_cmd =r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
        element="C", pair_style="lcbop", keep_tmp_files=False
    )

    for k in range(lenT):
        out = optimize_scalar_a_T_NPT(
            lattice_factory=make_diamond, a_init=a_guess, nx=nx, ny=ny, nz=nz, potential=pot,T_K=T[k], P_bar=1.0,dt_fs=1.0, total_steps=30000, equil_steps=10000,block_steps=100, seed=12345,tau_t_ps=0.1, tau_p_ps=1.0,
            keep_tmp_files=False, verbose=True
            )
        print("P_bar=",out["P_bar"])

        #C_GPa =build_C_by_central_difference(
        #    pot, pos0=out["pos_avg"],cell0=out["cell_avg"], temperature_K=T[k],
        #    dt_fs=1.0, total_steps=30000, block_steps=100,
        #    equil_steps=4000, strain_eps=2e-3, seed=12345,
        #    drop_frac=0.5,convention="engineering",
        #    eps_mags=[1e-3,2e-3], repeats=1, same_seed_pm=True,
        #    tdamp_ps=None, enforce_cubic=False
        #    )
        
        C_GPa = build_C_by_central_difference_NPT(
            pot, pos0=out["pos_avg"],cell0=out["cell_avg"], temperature_K=T[k],
            dt_fs=1.0, total_steps=20000, block_steps=100,
            equil_steps=2000, strain_eps=2e-3, seed=12345,
            drop_frac=0.5,convention="engineering",
            eps_mags=None, repeats=1, same_seed_pm=True,
            tdamp_ps=None, enforce_cubic=True
            )
        print(f"{T[k]}K","C (GPa) via central difference =\n", C_GPa)

def zero_K():
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
    #C0 = build_C_0K_via_scans(
    #    pos0=out['pos0'], cell0=out['cell0'], potential=pot,
    #    eps_max=2e-3, n_points=5, delta_fd=2e-4,
    #    convention="engineering", to_GPa=True,
    #    relax_params=dict(f_tol=1e-6, maxit=2000),
    #    method='external', symmetrize=True
    #)

    C0 = build_C_0K_central_difference(
        pos0=out['pos0'], cell0=out['cell0'], potential=pot,
        strain_eps=1e-3, delta_fd=2e-4, convention="engineering",
        relax_params=dict(f_tol=1e-6, maxit=2000),
        use_nlist=False, cutoff=None,
        volume_ref='reference', verbose=False, method='external',
        to_GPa=True, symmetrize=True
    )
    print("0 K: C (GPa) via central difference =\n", C0)

if __name__ == "__main__":
    T = [ 10 ]
    main(T)
    #zero_K()


