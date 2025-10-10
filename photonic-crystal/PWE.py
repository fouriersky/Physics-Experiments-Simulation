"""
pwe_tm.py -- 2D photonic crystal PWE (TM modes) demonstration
- Square lattice, circular dielectric rods in air
- Real-space grid used to compute Fourier components of 1/epsilon (more robust)
- Eigenproblem: sum_{G'} [(k+G)·(k+G')] * epsinv(G-G') H_{G'} = (ω/c)^2 H_G
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from scipy.linalg import eigh

# ---------- PARAMETERS ----------
a = 1.0               # lattice constant
eps_in = 12.0         # rod dielectric constant
eps_out = 1.0         # background dielectric
R = 0.2 * a           # rod radius
Ngrid = 64            # real-space grid resolution (use larger for accuracy)
Gmax_shell = 3        # integer shell for G = 2π(m,n)/a, m,n in [-Gmax_shell, Gmax_shell]
# Note: number of plane waves NG = (2*Gmax_shell+1)^2

# ---------- build real-space epsinv(r) and FFT to get epsinv_G ----------
x = np.linspace(-a/2, a/2, Ngrid, endpoint=False)
y = np.linspace(-a/2, a/2, Ngrid, endpoint=False)
XX, YY = np.meshgrid(x, y, indexing='xy')
rgrid = np.sqrt(XX**2 + YY**2)
eps_r = np.where(rgrid <= R, eps_in, eps_out)  
epsinv_r = 1.0 / eps_r

# FFT and shift so k=0 is center
epsinv_G_full = fftshift(fft2(epsinv_r)) / (Ngrid**2)

# grid of kx, ky values corresponding to FFT bins (useful to map G -> fft index)
kx_vals = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Ngrid, d=a/Ngrid))
ky_vals = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Ngrid, d=a/Ngrid))

# ---------- construct plane-wave G list ----------
Glist = []
Gindex = {}
idx = 0
for m in range(-Gmax_shell, Gmax_shell+1):
    for n in range(-Gmax_shell, Gmax_shell+1):
        G = (2*np.pi / a) * np.array([m, n])
        Glist.append(G)
        Gindex[(m, n)] = idx
        idx += 1
Glist = np.array(Glist)
NG = len(Glist)
print("Plane-wave count NG =", NG)

def G_to_fft_index(Gvec):
    """ map a G vector (kx,ky) to indices in the fft grid (iy,ix) """
    kx, ky = Gvec
    ix = np.argmin(np.abs(kx_vals - kx))
    iy = np.argmin(np.abs(ky_vals - ky))
    return iy, ix

# ---------- high-symmetry path (Γ-X-M-Γ) ----------
Gamma = np.array([0.0, 0.0])
X = np.array([np.pi/a, 0.0])
M = np.array([np.pi/a, np.pi/a])
n_k_seg = 30
def linpath(p1, p2, n):
    return [p1 + (p2-p1)*t for t in np.linspace(0,1,n,endpoint=False)]

path = []
path += linpath(Gamma, X, n_k_seg)
path += linpath(X, M, n_k_seg)
path += linpath(M, Gamma, n_k_seg+1)

# ---------- solve eigenproblem on each k ----------
bands = []
for kvec in path:
    Mmat = np.zeros((NG, NG), dtype=np.complex128)
    for i, G in enumerate(Glist):
        KGi = kvec + G
        for j, Gp in enumerate(Glist):
            KGj = kvec + Gp
            # Fourier component of epsinv at G-G'
            Gdiff = G - Gp
            iy, ix = G_to_fft_index(Gdiff)
            epsinv_Gdiff = epsinv_G_full[iy, ix]
            Mmat[i, j] = np.dot(KGi, KGj) * epsinv_Gdiff
    # Hermitian symmetrize and diagonalize
    M_sym = (Mmat + Mmat.conj().T) / 2.0
    evals, evecs = eigh(M_sym)
    omega_c = np.sqrt(np.maximum(np.real(evals), 0.0))   # (ω/c)
    bands.append(np.sort(omega_c))

bands = np.array(bands)

# ---------- plot first several bands ----------
num_bands_to_plot = min(8, NG)
n_kpoints = len(path)
plt.figure(figsize=(6,4.5))
for b in range(num_bands_to_plot):
    plt.plot(range(n_kpoints), bands[:, b], '-k', linewidth=1)
xticks = [0, n_k_seg, 2*n_k_seg, 3*n_k_seg]
plt.xticks(xticks, ['G','X','M','G'])
plt.ylabel(r'$\omega$ a / (2\pi c)  (normalized)')
plt.title('2D square-lattice PWE (TM modes)')
plt.grid(True, linestyle=':', alpha=0.4)
plt.savefig('./photonic-crystal/PWE-TM.png',dpi=300)
#plt.show()

# ---------- print Gamma-point eigenvalues (前几条) ----------
idx_gamma = 0
evals_gamma = bands[idx_gamma, :num_bands_to_plot]
print("Γ 点前几条带 (ω a normalized):")
for i, w in enumerate(evals_gamma):
    print(f"  Band {i+1}: {w:.6f}")

# ---------- 示例：用对称性判断简并（数值近似） ----------
# 在 Γ 点：如果两个本征值数值上几乎相等（在容差内），则视为简并。
tol = 1e-6
pairs = []
for i in range(len(evals_gamma)):
    for j in range(i+1, len(evals_gamma)):
        if abs(evals_gamma[i] - evals_gamma[j]) < 1e-6:
            pairs.append((i+1, j+1))
if pairs:
    print("在 Γ 点探测到近似简并的带对：", pairs)
else:
    print("未在列出的前若干带中检测到数值上严格近似的简并（小矩阵可能没有出现精确简并）。")
