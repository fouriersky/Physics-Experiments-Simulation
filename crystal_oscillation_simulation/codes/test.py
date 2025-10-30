if __name__ == "__main__":
    """
    calculate C-diamond stiffness tensor
    
    """
    from potential import DirectLAMMPSLCBOPPotential
    from opt_method import optimize_scalar_a_T_NPT,make_diamond
    from MD_module import build_C_by_central_difference
    nx, ny, nz = 2, 2, 2
    a_guess = 3.6
    
    pot = DirectLAMMPSLCBOPPotential(
        lcbop_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
        lmp_cmd =r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
        element="C", pair_style="lcbop", keep_tmp_files=False
    )
    out = optimize_scalar_a_T_NPT(
        lattice_factory=make_diamond, a_init=a_guess, nx=nx, ny=ny, nz=nz, potential=pot,T_K=300.0, P_bar=1.0,dt_fs=1.0, total_steps=20000, equil_steps=5000,block_steps=100, seed=12345,tau_t_ps=0.1, tau_p_ps=1.0,
        keep_tmp_files=False, verbose=True
        )

    #C_GPa = build_C_by_central_difference(
    #    backend=pot,
    #    pos0=out['pos_avg'], cell0=out['cell_avg'],
    #    temperature_K=300, dt_fs=1.0,
    #    total_steps=30000, block_steps=100, equil_steps=3000,
    #    strain_eps=1e-3, seed=12345, drop_frac=0.3,convention="engineering"
    #)
    #print("C (GPa) via central difference =\n", C_GPa)
