import numpy as np
from SK import SKParameters, SKTightBinding
from hopping import optical_eps2,plot_eps_spectra,plot_optical_constants,kk_epsilon1
from apply_strain import apply_strain


if __name__ == "__main__":

    params = SKParameters()
    model = SKTightBinding(params, basis="sp3d5s1")
    model.distance_for_scaling = None
    apply_strain(model, epsilon=0.02, axis='z', poisson_ratio=0.28)

    kgrid = model.compute_kgrid(n=15, gamma_centered=True, reduce="TR")
    omega = np.linspace(0.02, 40.0, 400)
    res = optical_eps2(model, kgrid,omega=omega, electrons_per_cell=8,                   
                       spin_degeneracy=2,eta=0.06,polarization=np.array([1,0,1]), window_valence_below=10, window_conduction_above=10)
    eps1 = kk_epsilon1(res['omega'], res['eps2'], add_one=True)
    #plot_eps_spectra(res['omega'], res['eps2'], title=f"ε (gap≈{res['gap']:.3f} eV)",filepath=None)
    plot_optical_constants(res['omega'], eps1, res['eps2'],title="Optical constants E_x_z", savepath='./pic/test_E_x_z.png')
