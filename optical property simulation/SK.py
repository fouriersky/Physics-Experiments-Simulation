import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

# 统一基组顺序（每原子 10 轨道，用于 sp3d5s*）
ORBITALS_10 = ['s','px','py','pz','dxy','dyz','dzx','dx2y2','dz2','s1']
ORBITALS_4  = ['s','px','py','pz']

# 14 (lm,v) combination of 10 orbitals: sp3d5s*
TWOCENTER_KEYS = (
    'sss','s1s1s','ss1s','sps','s1ps','sds','s1ds',
    'pps','ppp','pds','pdp','dds','ddp','ddd'
)


class DistanceScaler:
    """
    距离缩放：V(d) = V0 * (d0/d)^n 或 V0 * exp[-alpha (d/d0 - 1)]
    """
    def __init__(self, d0: float, mode: str = "power", power_n: float = 2.0, alpha: float = 2.0):
        self.d0 = d0
        self.mode = mode
        self.power_n = power_n
        self.alpha = alpha

    def __call__(self, V0: float, d: float) -> float:
        if V0 == 0.0:
            return 0.0
        if self.mode == "power":
            return V0 * (self.d0 / d)**self.power_n
        elif self.mode == "exp":
            return V0 * np.exp(-self.alpha * (d / self.d0 - 1.0))
        else:
            return V0

@dataclass
class SKParameters:
    """
    Si-Si Slate-Koster parameters, nearest-neighbor-only
    - onsite: E_s, E_p, E_d, E_s1
    - two_center: 14 two-center integral parameters
    - offer 2 sets of SK parameters, from 2 article respectively: 
        PhysRevB.57.6493, PhysRevB.69.115201
    """

    a: float = 5.431
    # PhysRevB.57.6493
    #E_s: float = -2.0196
    #E_p: float =  4.5448
    #E_d: float =  14.1836
    #E_s1: float =  19.6748
    
    # PhysRevB.69.115201
    E_s: float = -2.15168
    E_p: float =  4.22925
    E_d: float = 13.78950
    E_s1: float = 19.11650
    
    two_center: Dict[str, float] = field(default_factory=lambda: {
        # PhysRevB.57.6493
        #'sss':  -1.9413,   
        #'s1s1s': -3.3081,  
        #'ss1s': -1.6933,   
        #'sps':   2.7836,   
        #'s1ps':  2.8428,   
        #'sds':  -2.7998,   
        #'s1ds': -0.7003,   
        #'pps':   4.1068,   
        #'ppp':  -1.5934,   
        #'pds':  -2.1073,   
        #'pdp':   1.9977,   
        #'dds':  -1.2327,   
        #'ddp':   2.5145,   
        #'ddd':  -2.4734,  
        

        # PhysRevB.69.115201
        'sss':  -1.95933,
        's1s1s': -4.24135,
        'ss1s': -1.52230,
        'sps':   3.02562,
        's1ps':  3.15565,
        'sds':  -2.28485,
        's1ds': -0.80993,
        'pps':   4.10364,
        'ppp':  -1.51801,
        'pds':  -1.35554,
        'pdp':   2.38479,
        'dds':  -1.68136,
        'ddp':   2.58880,
        'ddd':  -1.81400,
    })

    # 距离缩放参数
    scale_mode: str = "power"
    power_n: float = 2.0
    alpha: float = 2.0
    d0: Optional[float] = None

    def __post_init__(self):
        # 校验 two_center keys 完整且无多余
        missing = [k for k in TWOCENTER_KEYS if k not in self.two_center]
        extra   = [k for k in self.two_center.keys() if k not in TWOCENTER_KEYS]
        if missing:
            raise ValueError(f"two_center 缺少键: {missing}")
        if extra:
            raise ValueError(f"two_center 包含未识别键: {extra}")
        # 校验值类型
        for k, v in self.two_center.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"two_center['{k}'] 必须为数值，得到 {type(v)}")

        # 计算最近邻参考距离
        d0_val = self.d0 if self.d0 is not None else float(np.sqrt(3.0) * self.a / 4.0)
        self.scaler = DistanceScaler(d0_val, mode=self.scale_mode, power_n=self.power_n, alpha=self.alpha)

    def scaled_params_at_distance(self, d: float) -> Dict[str, float]:
        return {k: self.scaler(v, d) for k, v in self.two_center.items()}
    
class DiamondLattice:
    """
    金刚石两原子单胞（菱方60°原胞）+ 最近邻 A->B 位移集合
    """
    def __init__(self, a: float):
        self.a = a
        # 原胞实空间基矢（菱方60°表示）
        self.a1 = np.array([0.0, a/2, a/2])
        self.a2 = np.array([a/2, 0.0, a/2])
        self.a3 = np.array([a/2, a/2, 0.0])
        # 两个子格坐标
        self.rA = np.array([0.0, 0.0, 0.0])
        self.rB = np.array([a/4, a/4, a/4])
        # 最近邻从 A 指向 B 的四个 R
        self.nn_R = np.array([
            [ a/4,  a/4,  a/4],
            [ a/4, -a/4, -a/4],
            [-a/4,  a/4, -a/4],
            [-a/4, -a/4,  a/4]
        ])
        # 逆格矢
        V = np.dot(self.a1, np.cross(self.a2, self.a3))
        self.b1 = 2*np.pi * np.cross(self.a2, self.a3) / V
        self.b2 = 2*np.pi * np.cross(self.a3, self.a1) / V
        self.b3 = 2*np.pi * np.cross(self.a1, self.a2) / V

    def reciprocal_matrix(self):
        return np.vstack([self.b1, self.b2, self.b3])

    @staticmethod
    def direction_cosines(R: np.ndarray) -> Tuple[float, float, float, float]:
        d = float(np.linalg.norm(R))
        if d == 0.0:
            raise ValueError("R 向量为零。")
        l, m, n = (R / d).tolist()
        return l, m, n, d


def sk_block_sp3(l: float, m: float, n: float, V: Dict[str, float]) -> np.ndarray:
    """
    返回 4x4 两中心子块 (s, px, py, pz) for 最近邻
    V: 仅用到 'sss','sps','pps','ppp'
    注意：
    - 这是“定向”的 AB 跃迁子块，并非厄米矩阵；整体 H(k) 由
      H_AB(R) e^{ik·R} 与 H_BA(-R)=H_AB(R).T e^{-ik·R} 组装后才厄米。
    - s↔p 属于 σ 型并在 AB 子块内成对取反：
        H_AB(s, p_i) = l_i V_{sps}
        H_AB(p_i, s) = -l_i V_{sps}
      反号源自 p 轨道沿键方向的奇宇称与 SK 表约定，等价于 R→-R。
    - p↔p 子块按 SK：H_{p_i p_j} = l_i l_j (Vppσ - Vppπ) + δ_{ij} Vppπ，
      因而在 AB 子块内是对称的。
    """
    H = np.zeros((4,4), dtype=float)
    # s-s σ
    H[0,0] = V['sss']
    # s-p σ
    H[0,1] = l * V['sps']
    H[0,2] = m * V['sps']
    H[0,3] = n * V['sps']
    H[1,0] = -H[0,1]
    H[2,0] = -H[0,2]
    H[3,0] = -H[0,3]
    # p-p
    Vpps = V['pps']
    Vppp = V['ppp']
    lvec = [l,m,n]
    for i in range(3):
        for j in range(3):
            val = lvec[i]*lvec[j]*(Vpps - Vppp)
            if i == j:
                val += Vppp
            H[1+i, 1+j] = val
    return H

def full_block_sp3d5s1(l: float, m: float, n: float, V: Dict[str, float]) -> np.ndarray:
    """
    构造 10x10 最近邻两中心子块 H_AB(R) (A->B; 方向余弦 l,m,n)
    轨道顺序: [s, px, py, pz, dxy, dyz, dzx, dx2y2, dz2, s1]
    说明:
    - s/p 与 d 为偶–偶/奇–偶的常规 SK 约定：
        s(s1)↔d 对称复制;  p↔d 反号复制
    - p↔d 多项式按 x→y→z 的循环置换严格补全，避免缺项/错项。
    """
    H = np.zeros((10,10), dtype=float)

    # ---- 预计算 ----
    l2, m2, n2 = l*l, m*m, n*n
    sqrt3 = np.sqrt(3.0)

    # ---- 先嵌入 sp3 ----
    H[:4,:4] = sk_block_sp3(l, m, n, V)

    # ---- s1-s1、s-s1 ----
    H[0,9] = V['ss1s']
    H[9,0] = V['ss1s']
    H[9,9] = V['s1s1s']

    # ---- s ↔ d (偶-偶, 对称) ----
    Vsds = V['sds']
    H[0,4] = sqrt3 * l * m * Vsds          # s-dxy
    H[0,5] = sqrt3 * m * n * Vsds          # s-dyz
    H[0,6] = sqrt3 * n * l * Vsds          # s-dzx
    H[0,7] = (sqrt3/2.0) * (l2 - m2) * Vsds  # s-dx2y2
    H[0,8] = (n2 - 0.5*(l2 + m2)) * Vsds     # s-dz2
    # 对称复制
    H[4,0] = H[0,4]; H[5,0] = H[0,5]; H[6,0] = H[0,6]
    H[7,0] = H[0,7]; H[8,0] = H[0,8]

    # ---- s1 ↔ d (偶-偶, 对称) ----
    Vs1ds = V['s1ds']
    H[9,4] = sqrt3 * l * m * Vs1ds
    H[9,5] = sqrt3 * m * n * Vs1ds
    H[9,6] = sqrt3 * n * l * Vs1ds
    H[9,7] = (sqrt3/2.0) * (l2 - m2) * Vs1ds
    H[9,8] = (n2 - 0.5*(l2 + m2)) * Vs1ds
    # 对称复制
    H[4,9] = H[9,4]; H[5,9] = H[9,5]; H[6,9] = H[9,6]
    H[7,9] = H[9,7]; H[8,9] = H[9,8]

    # ---- s1 ↔ p (奇偶, 反号) ----
    Vs1ps = V['s1ps']
    H[9,1] = l * Vs1ps
    H[9,2] = m * Vs1ps
    H[9,3] = n * Vs1ps
    H[1,9] = -H[9,1]; H[2,9] = -H[9,2]; H[3,9] = -H[9,3]

    # ---- p ↔ d (奇偶, 反号) ----
    Vpds = V['pds']
    Vpdπ = V['pdp']

    # px 行
    H[1,4] = sqrt3*l2*m*Vpds + m*(1-2*l2)*Vpdπ
    H[1,5] = sqrt3*l*m*n*Vpds - 2.0*l*m*n*Vpdπ
    H[1,6] = sqrt3*l2*n*Vpds + n*(1-2*l2)*Vpdπ

    H[1,7] = 0.5*sqrt3*l*(l2 - m2)*Vpds + l*(1 - l2 + m2)*Vpdπ
    H[1,8] = (l*n2 - 0.5*l*(l2 + m2))*Vpds - sqrt3*l*n2*Vpdπ
    # py 行
    H[2,4] = sqrt3*l*m2*Vpds + l*(1-2*m2)*Vpdπ                            
    H[2,5] = sqrt3*n*m2*Vpds + n*(1-2*m2)*Vpdπ          
    H[2,6] = sqrt3*l*m*n*Vpds - 2.0*l*m*n*Vpdπ                     

    H[2,7] = 0.5*sqrt3*m*(l2 - m2)*Vpds - m*(1 + l2 - m2)*Vpdπ
    H[2,8] = (m*n2 - 0.5*m*(m2 + l2))*Vpds - sqrt3*m*n2*Vpdπ
    # pz 行
    H[3,4] = sqrt3*l*m*n*Vpds - 2.0*l*m*n*Vpdπ                           
    H[3,5] = sqrt3*m*n2*Vpds + m*(1-2*n2)*Vpdπ                     
    H[3,6] = sqrt3*l*n2*Vpds + l*(1-2*n2)*Vpdπ         

    H[3,7] = 0.5*sqrt3*n*(l2 - m2)*Vpds - n*(l2 - m2)*Vpdπ
    H[3,8] = (n*n2 - 0.5*n*(m2 + l2))*Vpds + sqrt3*n*(m2+l2)*Vpdπ

    # 反号复制：d even, p odd
    for p_idx in [1,2,3]:
        for d_idx in [4,5,6,7,8]:
            H[d_idx, p_idx] = -H[p_idx, d_idx]

    # ---- d ↔ d (偶-偶, 对称) ----
    Vdds = V['dds']; Vddπ = V['ddp']; Vddδ = V['ddd']
    # 对角
    H[4,4] = 3*l2*m2*Vdds + (l2 + m2 - 4*l2*m2)*Vddπ + (n2 + l2*m2)*Vddδ
    H[5,5] = 3*m2*n2*Vdds + (m2 + n2 - 4*m2*n2)*Vddπ + (l2 + m2*n2)*Vddδ
    H[6,6] = 3*n2*l2*Vdds + (n2 + l2 - 4*n2*l2)*Vddπ + (m2 + n2*l2)*Vddδ
    # 非对角
    H[4,5] = 3*l*m2*n*Vdds + (l*n*(1 - 4*m2))*Vddπ + (l*n*(m2 - 1))*Vddδ
    H[4,6] = 3*l2*m*n*Vdds + (m*n*(1 - 4*l2))*Vddπ + (m*n*(l2 - 1))*Vddδ 
    H[5,6] = 3*l*m*n2*Vdds + (l*m*(1 - 4*n2))*Vddπ + (l*m*(n2 - 1))*Vddδ

    # 与 dx2y2, dz2 的耦合
    diff_lm = (l2 - m2)
    H[4,7] = 1.5*l*m*diff_lm*Vdds - 2*l*m*diff_lm*Vddπ + 0.5*l*m*diff_lm*Vddδ
    H[4,8] = sqrt3*l*m*(n2-0.5*(l2+m2))*Vdds - 2*sqrt3*l*m*n2*Vddπ + 0.5*sqrt3*l*m*(1+n2)*Vddδ
    H[5,7] = 1.5*m*n*diff_lm*Vdds - m*n*(1 + 2*diff_lm)*Vddπ + m*n*(1+0.5*diff_lm)*Vddδ
    H[5,8] = sqrt3*n*m*(n2-0.5*(l2+m2))*Vdds + sqrt3*m*n*(1-2*n2)*Vddπ - 0.5*sqrt3*m*n*(1-n2)*Vddδ
    H[6,7] = 1.5*n*l*diff_lm*Vdds + n*l*(1 - 2*diff_lm)*Vddπ - n*l*(1-0.5*diff_lm)*Vddδ
    H[6,8] = sqrt3*l*n*(n2-0.5*(l2+m2))*Vdds + sqrt3*l*n*(1-2*n2)*Vddπ - 0.5*sqrt3*l*n*(1-n2)*Vddδ 
    H[7,7] = 0.75*diff_lm**2*Vdds + ((l2 + m2) - diff_lm**2)*Vddπ + (n2 + 0.25*diff_lm**2)*Vddδ
    H[7,8] = (sqrt3/2.0)*diff_lm*(n2 - 0.5*(l2+m2))*Vdds - sqrt3*diff_lm*n2*Vddπ + (sqrt3/4.0)*diff_lm*(1 + n2)*Vddδ
    H[8,8] = ((3*n2 - 1)**2/4.0)*Vdds + 3*n2*(1 - n2)*Vddπ + 0.75*((1 - n2)**2)*Vddδ

    # 对称复制 (d-d)
    for i in range(4,9):
        for j in range(i+1,9):
            H[j,i] = H[i,j]

    return H

class SKDirectionPolynomials:
    """
    方向余弦多项式提供者。
    """
    def __init__(self):
        self.sp3_block_builder: Callable[[float,float,float,Dict[str,float]], np.ndarray] = sk_block_sp3
        self.sp3d5s1_block_builder: Callable[[float,float,float,Dict[str,float]], np.ndarray] = full_block_sp3d5s1

    def set_sp3d5s1_block_builder(self, func: Callable[[float,float,float,Dict[str,float]], np.ndarray]):
        self.sp3d5s1_block_builder = func


class SKTightBinding:
    """
    Slater-Koster TB 主类：构造实空间矩阵元与 H(k)
    """
    def __init__(self,
                 params: SKParameters,
                 basis: str = "sp3",                 # "sp3" 或 "sp3d5s1"
                 poly: Optional[SKDirectionPolynomials] = None,
                 distance_for_scaling: Optional[float] = None):
        self.params = params
        self.lattice = DiamondLattice(params.a)
        self.poly = poly if poly is not None else SKDirectionPolynomials()
        if basis not in ("sp3", "sp3d5s1"):
            raise ValueError("basis only supports 'sp3' or 'sp3d5s1'")
        self.basis = basis
        self.orbitals = ORBITALS_4 if basis == "sp3" else ORBITALS_10
        self.n_orb_atom = len(self.orbitals)
        self.nn_distance = np.linalg.norm(self.lattice.nn_R[0])
        self.distance_for_scaling = distance_for_scaling

    # ---------- 实空间两中心子块 ----------
    def two_center_block(self, R: np.ndarray) -> np.ndarray:
        """
        返回 H_AB(R) 子块(A->B),大小 n_orb_atom * n_orb_atom
        """
        l, m, n, d_geom = self.lattice.direction_cosines(R)
        d_use = self.distance_for_scaling if self.distance_for_scaling is not None else d_geom
        V_scaled = self.params.scaled_params_at_distance(d_use)
        if self.basis == "sp3":
            return self.poly.sp3_block_builder(l, m, n, V_scaled)
        else:
            return self.poly.sp3d5s1_block_builder(l, m, n, V_scaled)

    # ---------- 实空间矩阵元查询 ----------
    def H_realspace(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        返回最近邻 4 个 R 的 AB 子块，键为 (shell_index, R_index)
        仅最近邻：shell_index 固定为 1
        """
        blocks = {}
        for iR, R in enumerate(self.lattice.nn_R):
            blocks[(1, iR)] = self.two_center_block(R)
        return blocks

    # ---------- H(k) ----------
    def hamiltonian_k(self, k: np.ndarray) -> np.ndarray:
        nA = self.n_orb_atom
        H = np.zeros((2*nA, 2*nA), dtype=complex)
        # onsite
        for a in range(2):
            base = a*nA
            for i, orb in enumerate(self.orbitals):
                if orb == 's':
                    H[base+i, base+i] = self.params.E_s
                elif orb.startswith('p'):
                    H[base+i, base+i] = self.params.E_p
                elif orb.startswith('d'):
                    H[base+i, base+i] = self.params.E_d
                elif orb == 's1':
                    H[base+i, base+i] = self.params.E_s1
        # AB 最近邻 Bloch 和
        for R in self.lattice.nn_R:
            block = self.two_center_block(R)
            phase = np.exp(1j * np.dot(k, R))
            H[0:nA, nA:2*nA] += block * phase
            H[nA:2*nA, 0:nA] += block.T.conj() * np.conj(phase)  # 两中心仅依赖方向，AB=BA^T
        return H

    # ---------- band structure ----------
    def band_structure(self, k_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(k_path)
        nband = 2 * self.n_orb_atom
        evals = np.zeros((N, nband))
        evecs = np.zeros((N, nband, nband), dtype=complex)
        for i, k in enumerate(k_path):
            Hk = self.hamiltonian_k(k)
            w, v = np.linalg.eigh(Hk)
            evals[i] = w.real
            evecs[i] = v
        return evals, evecs

    def band_properties(self,
                        k_path: np.ndarray,
                        electrons_per_cell: Optional[int] = None,
                        spin_degeneracy: int = 2) -> Dict[str, Optional[float]]:
        """
        计算能带性质（无自旋假设）
        electrons_per_cell: 默认 2*4=8 (Si 两原子 4 价电子各)
        返回:
          dict 包含:
            evals        : (N, nbands) 本征值
            VBM          : 价带顶能量
            CBM          : 导带底能量 (可能为 None 若金属)
            gap          : 间隙 (负表示重叠)
            direct_gap   : min_k [E_c(k)-E_v(k)]
            VBM_k_index  : VBM 所在 k 索引
            CBM_k_index  : CBM 所在 k 索引 (None if 金属)
            is_indirect  : True/False/None
        """
        evals, _ = self.band_structure(k_path)
        N, nbands = evals.shape

        if electrons_per_cell is None:
            electrons_per_cell = 8  # Si: 两原子×4 价电子
        if spin_degeneracy <= 0:
            raise ValueError("spin_degeneracy 必须为正整数。")

        # 无自旋谱中应以自旋简并折算占据带数（Si -> 8/2 = 4 条带）
        occ_bands = int(np.floor(electrons_per_cell / spin_degeneracy))
        if occ_bands <= 0 or occ_bands > nbands:
            raise ValueError(f"占据带数={occ_bands} 不合理（总带数={nbands}）。")
        
        # 价带顶/导带底
        v_band = occ_bands - 1
        c_band = occ_bands if occ_bands < nbands else None

        VBM_vals = evals[:, v_band]
        VBM = float(np.max(VBM_vals))
        VBM_k_index = int(np.argmax(VBM_vals))

        if c_band is None:
            return dict(evals=evals, VBM=VBM, CBM=None, gap=None,
                        direct_gap=None, VBM_k_index=VBM_k_index,
                        CBM_k_index=None, is_indirect=None)

        CBM_vals = evals[:, c_band]
        CBM = float(np.min(CBM_vals))
        CBM_k_index = int(np.argmin(CBM_vals))
        gap = CBM - VBM
        direct_gap = float(np.min(CBM_vals - VBM_vals))
        is_indirect = (VBM_k_index != CBM_k_index) if gap >= 0 else None

        return dict(
            evals=evals,
            VBM=VBM,
            CBM=CBM,
            gap=gap,
            direct_gap=direct_gap,
            VBM_k_index=VBM_k_index,
            CBM_k_index=CBM_k_index,
            is_indirect=is_indirect
        )
    
    def plot_band_structure(self,
                            k_path: np.ndarray,
                            k_labels: Optional[list] = None,
                            k_positions: Optional[list] = None,
                            save_path: Optional[str] = None,
                            electrons_per_cell: Optional[int] = None,
                            ymin: Optional[float]=None, 
                            ymax: Optional[float]=None, 
                            spin_degeneracy: int = 2,
                            shift_mode: str = "none",
                            E_F: Optional[float] = None,
                            show_gap: bool = True):
        """
        shift_mode:
          none    : 不平移
          VBM     : VBM 置零
          midgap  : (VBM+CBM)/2 置零（若有隙）
          EF      : 使用参数 E_F（必须提供 E_F）
        """
        props = self.band_properties(k_path, electrons_per_cell)
        evals = props['evals']
        VBM = props['VBM']
        CBM = props['CBM']
        gap = props['gap']
        # 选择零点
        zero = 0.0
        if shift_mode == "VBM":
            zero = VBM
        elif shift_mode == "midgap":
            if CBM is not None and gap is not None and gap >= 0:
                zero = 0.5 * (VBM + CBM)
            else:
                zero = VBM
        elif shift_mode == "EF":
            if E_F is None:
                raise ValueError("shift_mode='EF', E_F required")
            zero = E_F
        shifted = evals - zero

        plt.figure(figsize=(8,6))
        for i in range(shifted.shape[1]):
            plt.plot(shifted[:, i], 'b-', lw=1.0)
        plt.axhline(0.0, color='red', ls='--', lw=0.8, label='reference')
        plt.xlabel('k-path'); plt.ylabel('Energy (eV)')
        title = f'Si TB ({self.basis})'
        if shift_mode != "none": title += f' | shift={shift_mode}'
        plt.title(title)
        if k_labels and k_positions:
            plt.xticks(k_positions, k_labels)
            for p in k_positions: plt.axvline(p, color='gray', ls='--', alpha=0.35)
        if show_gap and gap is not None:
            txt = f"Gap={gap:.3f} eV ({'indirect' if props['is_indirect'] else 'direct'})"
            plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
                     fontsize=9, va='top', ha='left', color='darkgreen')
        plt.legend(loc='lower right', fontsize=8)
        if ymax and ymin: plt.ylim(ymin,ymax)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        else: plt.show()

        print(f"VBM = {props['VBM']:.4f} eV @ k-index {props['VBM_k_index']}")
        if CBM is not None:
            print(f"CBM = {CBM:.4f} eV @ k-index {props['CBM_k_index']}")
            print(f"Indirect gap = {gap:.4f} eV (direct gap = {props['direct_gap']:.4f} eV)")
        else:
            print("No CBM: Maybe metallic or wrong! ")

    # ---------- k 网格生成（Monkhorst–Pack） ----------
    def _monkhorst_pack_fractional(self, n: int, gamma_centered: bool = True) -> np.ndarray:
        """
        生成 MP 网格的分数坐标（相对于 b1,b2,b3），范围在 [-0.5, 0.5)。
        gamma_centered=True 时，u = (i-(n-1)/2)/n，i=0..n-1；n 为奇数时包含 Γ。
        """
        if n <= 0:
            raise ValueError("n 必须为正整数")
        if gamma_centered:
            u = (np.arange(n) - (n - 1) / 2.0) / n
        else:
            # 标准 MP 位移：不一定包含 Γ
            u = (2.0 * (np.arange(1, n + 1)) - 1.0 - n) / (2.0 * n)
        kx, ky, kz = np.meshgrid(u, u, u, indexing='ij')
        frac = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)
        # wrap 到 [-0.5, 0.5)
        frac = ((frac + 0.5) % 1.0) - 0.5
        return frac

    @staticmethod
    def _wrap_frac(v: np.ndarray) -> np.ndarray:
        """把分数坐标 wrap 到 [-0.5, 0.5)"""
        return ((v + 0.5) % 1.0) - 0.5

    def _reduce_time_reversal(self, frac: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        """
        按时间反演对称性裁剪网格：
        - 只保留每对 {k, -k} 的“规范代表”(lexicographic 比较)
        - 返回: kept_frac, weights（Gamma 点权重=1，其余=2）
        说明：对当前 MP 网格，满足 k ≡ -k (mod G) 的只有 Γ（n 为奇数时）及少数边界点；
             我们采用 wrap 后的字典序比较确定代表。
        """
        kept = []
        weights = []
        for p in frac:
            q = self._wrap_frac(-p)
            # 判断是否自反 (k == -k)
            if np.allclose(p, q, atol=eps):
                # 自反：只保留一次，权重 1
                # 用一个简单规则：只有当三坐标都在 [-eps, eps] 近零时才记为自反
                if np.all(np.abs(p) < 1e-14):
                    kept.append(p)
                    weights.append(1.0)
                # 否则交给下面的字典序比较处理
                continue
            # 字典序比较，保留 (p >= q)
            take = False
            for a, b in zip(p, q):
                if np.isclose(a, b, atol=eps):
                    continue
                take = (a > b)
                break
            if take:
                kept.append(p)
                weights.append(2.0)
        kept = np.array(kept) if kept else np.zeros((0,3))
        weights = np.array(weights) if len(weights) else np.zeros((0,))
        # 如果 Γ 未包含（n 偶时），不会出现在 kept；这是正常的
        return kept, weights
    
    def compute_kgrid(self,
                      n: int,
                      gamma_centered: bool = True,
                      reduce: str = "TR",
                      return_fractional: bool = False) -> Dict[str, np.ndarray]:
        """
        生成 n×n×n 均匀网格并计算每个 k 点的本征值/本征矢。
        参数:
          - n: 每个方向的网格数
          - gamma_centered: 是否 Γ 居中
          - reduce: 'none'（全 BZ）或 'TR'（时间反演 IBZ 裁剪）
          - return_fractional: 是否同时返回分数坐标
        返回:
          dict 包含:
            'kpoints_cart' : (Nk,3) 倒易笛卡尔坐标(1/Å)
            'kpoints_frac' : (Nk,3) 分数坐标(可选)
            'weights'      : (Nk,) 网格权重（'none' 时全 1；'TR' 时 1 或 2）
            'evals'        : (Nk, nbands)
            'evecs'        : (Nk, nbands, nbands)
        注意:
          - 当前 reduce='TR' 仅使用时间反演对称性；完整点群 IBZ 可后续扩展。
        """
        # 1) 生成分数坐标 MP 网格
        frac_full = self._monkhorst_pack_fractional(n, gamma_centered=gamma_centered)
        # 2) 裁剪
        if reduce is None or reduce.lower() == "none":
            frac_kept = frac_full
            weights = np.ones((frac_kept.shape[0],), dtype=float)
        elif reduce.upper() == "TR":
            frac_kept, weights = self._reduce_time_reversal(frac_full)
            if frac_kept.size == 0:
                # 若全部被过滤（例如 n 很小且无 Γ），退化为不裁剪
                frac_kept = frac_full
                weights = np.ones((frac_kept.shape[0],), dtype=float)
        else:
            raise ValueError("reduce 仅支持 'none' 或 'TR'（时间反演）")

        # 3) 转笛卡尔
        B = self.lattice.reciprocal_matrix()
        k_cart = frac_kept @ B

        # 4) 对每个 k 计算本征系统
        evals_list = []
        evecs_list = []
        for k in k_cart:
            Hk = self.hamiltonian_k(k)
            w, v = np.linalg.eigh(Hk)
            evals_list.append(w.real)
            evecs_list.append(v)
        evals = np.array(evals_list)
        evecs = np.array(evecs_list)

        out = {
            'kpoints_cart': k_cart,
            'weights': weights,
            'evals': evals,
            'evecs': evecs
        }
        if return_fractional:
            out['kpoints_frac'] = frac_kept
        return out
    
def get_high_symmetry_path(
    a: float,
    n_points: int | list[int] = 50,
    points: Optional[Dict[str, np.ndarray]] = None,
    segments: Optional[list[tuple[str, str]]] = None,
    coords: str = "fractional",   # 'fractional' 基于 (b1,b2,b3) 的分数坐标；'cartesian' 直接给倒易笛卡尔坐标(1/Å)
    include_end: bool = True
) -> Tuple[np.ndarray, list, list]:
    """
    生成任意自定义高对称路径的 k 点序列。
    参数:
      - a: 晶格常数(Å)，用于构建倒易基矢。
      - n_points: 每段采样点数（含两端）；可为单个整数或与 segments 等长的列表。
      - points: 高对称点字典 {'Γ': frac_or_cart_array, ...}
      - segments: 路径片段列表 [('Γ','X'), ('X','U'), ...]
      - coords: 'fractional' 表示 points 中坐标为相对于 (b1,b2,b3) 的分数；'cartesian' 表示已是倒易笛卡尔坐标(1/Å)。
      - include_end: 是否每段包含终点（默认 True）。段与段之间会自动去重起点。
    返回:
      - k_path: (N,3) 倒易空间笛卡尔坐标(1/Å)
      - labels: 轴刻度标签
      - positions: 每个标签对应的索引位置
    """
    lat = DiamondLattice(a)
    B = lat.reciprocal_matrix()

    # 默认高对称点与路径（与原实现兼容）
    if points is None:
        points = {
            'Γ': np.array([0.0,   0.0,   0.0]),
            'X': np.array([0.5,   0.0,   0.5]),
            'U': np.array([0.625, 0.25,  0.625]),
            'K': np.array([0.375, 0.375, 0.75]),
            'L': np.array([0.5,   0.5,   0.5]),
            'W': np.array([0.5,   0.25,  0.75]),
        }
        default_segments = [('L','Γ'), ('Γ','X'), ('X','U'), ('K','Γ')]
        if segments is None:
            segments = default_segments

    # 坐标转换为倒易笛卡尔
    def to_cart(v):
        return v @ B if coords == "fractional" else v

    # n_points 归一化
    if isinstance(n_points, int):
        nplist = [n_points]*len(segments)
    else:
        if len(n_points) != len(segments):
            raise ValueError("n_points 列表长度必须等于 segments 长度")
        nplist = n_points

    k_list: list[np.ndarray] = []
    labels: list[str] = []
    positions: list[int] = []

    labels.append(segments[0][0])
    positions.append(0)
    total = 0

    for si, ((A, Bpt), npseg) in enumerate(zip(segments, nplist)):
        if A not in points or Bpt not in points:
            raise ValueError(f"未在 points 中找到 {A} 或 {Bpt}")
        kA = to_cart(points[A])
        kB = to_cart(points[Bpt])
        if npseg < 2:
            raise ValueError("每段 n_points 必须 >= 2")
        steps = npseg if include_end else (npseg - 1)
        for i in range(steps):
            t = i / (npseg - 1)
            k = (1.0 - t)*kA + t*kB
            # 段间去重: 除首段外，跳过第一个点
            if si > 0 and i == 0:
                continue
            k_list.append(k)
            total += 1
        labels.append(Bpt)
        positions.append(total - 1)

    return np.array(k_list), labels, positions





if __name__ == "__main__":
    params = SKParameters()
    model = SKTightBinding(params, basis="sp3d5s1")
    #  Γ-X-W-K-Γ-L path
    points = {
        'Γ': np.array([0.0, 0.0, 0.0]),
        'X': np.array([0.5, 0.0, 0.5]),
        'W': np.array([0.5, 0.25, 0.75]),
        'K': np.array([0.375, 0.375, 0.75]),
        'L': np.array([0.5, 0.5, 0.5]),
        'U': np.array([0.625, 0.25, 0.625])  # Added U point
    }

    # Path based on the band structure image
    segs = [
        ('L', 'Γ'),
        ('Γ', 'X'),
        ('X', 'W'),
        ('W', 'K'),
        ('K', 'L'),
        ('L', 'W'),
        ('W', 'X'),
        ('X', 'U'),
        ('K', 'Γ')
    ]
    k_path, labels, pos = get_high_symmetry_path(
        a=5.431,
        n_points=[40, 40, 40, 40, 40, 40, 40, 40, 40],
        points=points,
        segments=segs,
        coords='fractional',
        include_end=True
    )
    model.plot_band_structure(k_path, k_labels=labels, k_positions=pos,           save_path='./test.png',electrons_per_cell=8, ymin=-10, ymax=10, spin_degeneracy=2, shift_mode="midgap")

    props = model.band_properties(k_path, electrons_per_cell=8)
    print(props['gap'], props['direct_gap'], props['is_indirect'])
