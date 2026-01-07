import numpy as np
from typing import Dict, Tuple, Optional

ħ = 6.582119569e-16        # eV·s
m_e = 9.10938356e-31        # kg
e_charge = 1.602176634e-19  # C
HBAR_SI = 1.054571817e-34   # J·s
EPS0_SI = 8.8541878128e-12  # F/m
ANG2M   = 1.0e-10           # Å -> m
A3_TO_M3 = 1.0e-30          # Å^3 -> m^3
h_SI   = 2.0 * np.pi * HBAR_SI   # Planck 常数 J·s
c_SI   = 2.99792458e8            # 光速 m/s


def h_derivatives(model, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析计算 ∂H/∂kα (α = x,y,z)，返回三个 (Nb,Nb) 复矩阵。
    H(k) = H_onsite + Σ_R [ B e^{ik·R} + B^T e^{-ik·R} ]
    其中 B 为 A->B 两中心子块嵌入。
    """
    nA = model.n_orb_atom
    dim = 2 * nA
    dHx = np.zeros((dim, dim), dtype=complex)
    dHy = np.zeros((dim, dim), dtype=complex)
    dHz = np.zeros((dim, dim), dtype=complex)

    for R in model.lattice.nn_R:
        block = model.two_center_block(R)  # (nA, nA)
        phase = np.exp(1j * np.dot(k, R))
        Rx, Ry, Rz = map(float, R)
        # ∂/∂kα e^{ik·R} = i Rα e^{ik·R}
        dHx[0:nA, nA:2*nA] += (1j * Rx * phase)        * block
        dHy[0:nA, nA:2*nA] += (1j * Ry * phase)        * block
        dHz[0:nA, nA:2*nA] += (1j * Rz * phase)        * block
        # 反向 BA: ∂/∂kα e^{-ik·R} = -i Rα e^{-ik·R}
        dHx[nA:2*nA, 0:nA] += (-1j * Rx * np.conj(phase)) * block.T.conj()
        dHy[nA:2*nA, 0:nA] += (-1j * Ry * np.conj(phase)) * block.T.conj()
        dHz[nA:2*nA, 0:nA] += (-1j * Rz * np.conj(phase)) * block.T.conj()
    return dHx, dHy, dHz


def velocity_matrix_elements(evecs_k: np.ndarray,
                             dHks: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定某 k 点本征矢量矩阵 U (列为本征态) 与 dH/dkα，
    计算 vα = (1/ħ) U† (dH/dkα) U
    返回 (vx, vy, vz) 三个 (Nb,Nb) 复矩阵。
    """
    U = evecs_k
    Udag = U.conj().T
    dHx, dHy, dHz = dHks
    vx = (Udag @ dHx @ U) / ħ
    vy = (Udag @ dHy @ U) / ħ
    vz = (Udag @ dHz @ U) / ħ
    return vx, vy, vz


def kk_epsilon1(omega: np.ndarray, eps2: np.ndarray, add_one: bool = True) -> np.ndarray:
    """
    Kramers–Kronig: ε1(ω) = 1 + (2/π) P ∫_0^∞ [Ω ε2(Ω)] / (Ω^2 - ω^2) dΩ
    离散实现（去除对角项做主值），O(N^2)；适用于中等频率网格。
    说明:
      - omega, eps2 为实数组；omega 单位 eV（视为 ħω）。
      - 频率网格有限会有截断误差，可通过扩展网格或尾部拟合改善。
    """
    omega = np.asarray(omega, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    N = omega.size
    if N < 2:
        return np.ones_like(eps2) if add_one else np.zeros_like(eps2)
    # 梯形权重 w_j
    dω = np.diff(omega)
    w = np.empty(N)
    w[0] = dω[0] / 2.0
    w[-1] = dω[-1] / 2.0
    if N > 2:
        w[1:-1] = (dω[:-1] + dω[1:]) / 2.0
    F = omega * eps2 * w  # 加权后的 Ω ε2(Ω) * 权重
    # 构造分母矩阵 (ω_j^2 - ω_i^2)
    denom = omega[np.newaxis, :]**2 - omega[:, np.newaxis]**2  # shape (N,N)
    # 主值：去除对角项
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.zeros_like(denom)
        mask = ~np.isclose(denom, 0.0)
        kernel[mask] = 1.0 / denom[mask]
    # ε1_i = (2/π) Σ_j F_j / (ω_j^2 - ω_i^2)
    eps1 = (2.0 / np.pi) * (kernel @ F)
    if add_one:
        eps1 += 1.0
    return eps1

def optical_eps2(model,
                 kgrid: Dict[str, np.ndarray],
                 electrons_per_cell: int = 8,
                 spin_degeneracy: int = 2,
                 omega: Optional[np.ndarray] = None,
                 eta: float = 0.05,
                 polarization: np.ndarray = np.array([1.0, 0.0, 0.0]),
                 window_valence_below: float = 10.0,
                 window_conduction_above: float = 10.0,
                 normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    计算 ε2(ω)（物理量纲，velocity gauge）。
    改进：使用高斯展宽代替洛伦兹展宽，消除低频(ω < gap)处的虚假长尾信号。
    """
    kpts   = kgrid['kpoints_cart']
    weights= kgrid['weights']
    evals  = kgrid['evals']
    evecs  = kgrid['evecs']
    Nk, nb = evals.shape

    # 1. 晶胞体积 (SI: m^3)
    B = model.lattice.reciprocal_matrix()
    detB = float(np.linalg.det(B))
    Vcell_A3 = (2*np.pi)**3 / abs(detB)
    Vcell_SI = Vcell_A3 * A3_TO_M3

    # 2. 物理前因子
    # ε2 = (2π ħ^2) / (ε0 Ω e) * Σ |v|^2 * δ(E) / E^2
    K_pref = (2.0 * np.pi * (HBAR_SI**2)) / (EPS0_SI * Vcell_SI * e_charge)

    occ_bands = int(np.floor(electrons_per_cell / spin_degeneracy))
    v_max = occ_bands - 1
    c_min = occ_bands
    Nv = occ_bands
    Nc = nb - occ_bands
    if Nv <= 0 or Nc <= 0:
        raise ValueError("占据带或导带数量为 0。")

    VBM = float(np.max(evals[:, v_max]))
    CBM = float(np.min(evals[:, c_min]))
    gap = CBM - VBM
    if gap < 0:
        print(f"[警告] 金属态(gap={gap:.3f} eV)。")

    Ev_min = VBM - float(window_valence_below)
    Ec_max = CBM + float(window_conduction_above)

    # 3. 频率网格
    if omega is None:
        Emin = max(1e-2, gap if gap > 0 else 1e-2)
        Emax = min((Ec_max - Ev_min)*1.02,
                   np.max(evals[:, c_min:] - evals[:, [v_max]])*1.05)
        omega = np.linspace(Emin, Emax, 1000)
    omega = np.asarray(omega, dtype=float)

    ehat = polarization / np.linalg.norm(polarization)

    # 4. 收集数据
    all_dE_list = []
    all_M2_list = []
    all_w_list  = []

    for ik in range(Nk):
        w_k = weights[ik]
        Ev = evals[ik, :occ_bands].reshape(1, Nv)
        Ec = evals[ik, c_min:].reshape(Nc, 1)

        val_mask = (Ev >= Ev_min) & (Ev <= VBM)
        con_mask = (Ec >= CBM) & (Ec <= Ec_max)
        if not (np.any(val_mask) and np.any(con_mask)):
            continue
        mask = con_mask & val_mask

        dHks = h_derivatives(model, kpts[ik])
        vx, vy, vz = velocity_matrix_elements(evecs[ik], dHks)
        v_cv = (ehat[0]*vx[c_min:, :occ_bands] +
                ehat[1]*vy[c_min:, :occ_bands] +
                ehat[2]*vz[c_min:, :occ_bands])
        
        # 单位转换: Å/s -> m/s
        v_cv_SI = v_cv * ANG2M 
        
        M2 = np.abs(v_cv_SI)**2
        M2 = np.where(mask, M2, 0.0)

        dE = (Ec - Ev)
        sel = M2 > 1e-25
        if np.any(sel):
            all_dE_list.append(dE[sel])
            all_M2_list.append(M2[sel])
            all_w_list.append(np.full(np.count_nonzero(sel), w_k, dtype=float))

    if not all_dE_list:
        return {'omega': omega, 'eps2': np.zeros_like(omega), 'gap': gap, 'VBM': VBM, 'CBM': CBM}

    all_dE = np.concatenate(all_dE_list)
    all_M2 = np.concatenate(all_M2_list)
    all_w  = np.concatenate(all_w_list)

    # 5. 频率卷积 (改为高斯展宽)
    # Gaussian: (1 / (eta * sqrt(pi))) * exp( - ( (dE - w) / eta )^2 )
    diff = all_dE[:, None] - omega[None, :]
    
    # 使用高斯函数代替洛伦兹函数
    # 高斯函数在 |x| > 3*eta 时迅速衰减至 0，避免了 1/omega^2 放大低频尾部的问题
    gaussian = (1.0 / (eta * np.sqrt(np.pi))) * np.exp( - (diff / eta)**2 )

    # 6. 求和
    omega_safe_sq = np.maximum(omega**2, 1e-6)
    S_omega = np.sum(all_w[:, None] * all_M2[:, None] * gaussian / omega_safe_sq[None, :], axis=0)

    # 物理截断：再次确保极低频为 0 (可选，高斯展宽下通常已自动满足)
    S_omega[np.abs(omega) < 1e-3] = 0.0

    # 7. 权重归一化
    total_weight = np.sum(weights)
    if total_weight > 0:
        S_omega /= total_weight

    eps2 = K_pref * S_omega

    return {'omega': omega, 'eps2': eps2, 'gap': gap, 'VBM': VBM, 'CBM': CBM}

def plot_eps_spectra(omega: np.ndarray,
                     eps2: np.ndarray,
                     title: str = "Dielectric function",
                     filepath: Optional[str] = None):
    import matplotlib.pyplot as plt
    eps1 = kk_epsilon1(omega, eps2, add_one=True)
    plt.figure(figsize=(8,5))
    plt.plot(omega, eps1, 'r-', lw=1.1, label='ε1 (KK)')
    plt.plot(omega, eps2, 'b-', lw=1.1, label='ε2')
    plt.xlabel("ħω (eV)")
    plt.ylabel("ε (arb.)")
    plt.xlim(0, omega[-1])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.show()

def compute_optical_constants(omega_eV: np.ndarray,
                              eps1: np.ndarray,
                              eps2: np.ndarray) -> Dict[str, np.ndarray]:
    """
    计算 n(ω), k(ω), α(ω).
    输入:
      omega_eV : ħω (eV) (即光子能量)
      eps1, eps2 : 介电函数实部/虚部
    返回:
      dict: n, k, alpha_m (1/m), alpha_cm (1/cm)
    """
    omega_eV = np.asarray(omega_eV, dtype=float)
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    rad = np.sqrt(eps1*eps1 + eps2*eps2)
    n = np.sqrt((rad + eps1) * 0.5)
    k = np.sqrt(np.maximum((rad - eps1) * 0.5, 0.0))
    # α(E) = 4π k E / (h c)
    E_J = omega_eV * e_charge
    alpha_m = 4.0 * np.pi * k * E_J / (h_SI * c_SI)    # 1/m
    alpha_cm = alpha_m / 100.0
    return dict(n=n, k=k, alpha_m=alpha_m, alpha_cm=alpha_cm)

def plot_optical_constants(omega: np.ndarray,
                           eps1: np.ndarray,
                           eps2: np.ndarray,
                           title: str = "Optical constants",
                           savepath: Optional[str] = None):
    """
    绘制 n(ω), k(ω) 与 α(ω)。
    """
    import matplotlib.pyplot as plt
    consts = compute_optical_constants(omega, eps1, eps2)
    n = consts['n']; k = consts['k']; alpha_cm = consts['alpha_cm']
    fig, ax = plt.subplots(2, 1, figsize=(8,7), sharex=True)
    ax[0].plot(omega, n, 'g-', lw=1.2, label='n')
    ax[0].plot(omega, k, 'm-', lw=1.2, label='k')
    ax[0].set_xlim(0,omega[-1])
    ax[0].set_ylabel('n, k')
    ax[0].set_title(title)
    ax[0].legend()
    ax[1].plot(omega, alpha_cm, 'r-', lw=1.2, label=r'$\alpha$ (1/cm)')
    ax[1].set_xlabel(r'$\hbar \omega$ (eV)')
    ax[1].set_ylabel(r'$ \alpha $ (1/cm)')
    ax[1].set_xlim(0,omega[-1])
    ax[1].legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()
    return consts


if __name__=="__main__":
    from SK import SKParameters, SKTightBinding
    #from hopping import optical_eps2, plot_eps_spectra

    params = SKParameters()
    model = SKTightBinding(params, basis="sp3d5s1")
    kgrid = model.compute_kgrid(n=15, gamma_centered=True, reduce="TR")
    omega = np.linspace(0.02, 40.0, 400)
    res = optical_eps2(model, kgrid,omega=omega, electrons_per_cell=8,                   
                       spin_degeneracy=2,eta=0.06,polarization=np.array([0,1,0]), window_valence_below=10, window_conduction_above=10)
    eps1 = kk_epsilon1(res['omega'], res['eps2'], add_one=True)
    #plot_eps_spectra(res['omega'], res['eps2'], title=f"ε (gap≈{res['gap']:.3f} eV)",filepath='./pic/test_eps.png')
    plot_optical_constants(res['omega'], eps1, res['eps2'],title="Optical constants", savepath="./pic/optical_constants.png")
