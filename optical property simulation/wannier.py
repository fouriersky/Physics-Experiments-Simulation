import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
from SK import SKParameters, SKTightBinding

class WannierLocalizer:
    """
    在 SK-TB 模型基础上构建 sp3 杂化 Wannier 函数。
    对于金刚石结构 Si，价带可以通过投影到 4 个 sp3 键合轨道上获得很好的局域化。
    """
    def __init__(self, model: SKTightBinding):
        self.model = model
        if model.basis != "sp3" and model.basis != "sp3d5s1":
            raise ValueError("目前仅支持 sp3 或 sp3d5s1 基组进行 sp3 投影")
        
        # 定义 sp3 杂化系数 (对于四面体结构)
        # |h1> = 1/2 (|s> + |px> + |py> + |pz>)
        # |h2> = 1/2 (|s> + |px> - |py> - |pz>)
        # |h3> = 1/2 (|s> - |px> + |py> - |pz>)
        # |h4> = 1/2 (|s> - |px> - |py> + |pz>)
        self.sp3_coeffs = 0.5 * np.array([
            [1,  1,  1,  1],
            [1,  1, -1, -1],
            [1, -1,  1, -1],
            [1, -1, -1,  1]
        ])
        
        # 轨道索引映射 (SK.py 中 ORBITALS_10 = ['s','px','py','pz',...])
        self.orb_indices = [0, 1, 2, 3] # s, px, py, pz

    def get_sp3_hybrids_at_atom(self, atom_index: int) -> np.ndarray:
        """
        构建指定原子的 4 个 sp3 杂化轨道在 TB 基组下的表示向量。
        返回 shape (4, n_total_orbitals)
        """
        n_orb = self.model.n_orb_atom
        dim = 2 * n_orb
        hybrids = np.zeros((4, dim), dtype=complex)
        
        base = atom_index * n_orb
        for i in range(4): # 4 个 sp3 轨道
            for j, coeff in enumerate(self.sp3_coeffs[i]):
                # j=0->s, j=1->px, ...
                tb_idx = base + self.orb_indices[j]
                hybrids[i, tb_idx] = coeff
        return hybrids

    def compute_wannier_centers(self):
        """
        计算键合轨道中心。对于 Si，Wannier 中心应位于最近邻键的中点。
        """
        lat = self.model.lattice
        # 原子 A (0,0,0) 和 原子 B (a/4, a/4, a/4)
        rA = lat.rA
        rB = lat.rB
        
        # 键中心 (Bond Centers)
        centers = []
        # A 原子的 4 个键指向 B 的 4 个最近邻
        for R in lat.nn_R:
            # 键中点 = rA + R/2
            center = rA + 0.5 * R
            centers.append(center)
        return np.array(centers)

    def evaluate_orbital_spatial(self, 
                               coeffs: np.ndarray, 
                               grid_x: np.ndarray, 
                               grid_y: np.ndarray, 
                               grid_z: np.ndarray,
                               atom_pos: np.ndarray) -> np.ndarray:
        """
        在空间网格上计算波函数值。
        使用简单的 Slater-type Orbitals (STO) 近似径向部分以进行可视化。
        """
        # 相对坐标
        rx = grid_x - atom_pos[0]
        ry = grid_y - atom_pos[1]
        rz = grid_z - atom_pos[2]
        r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-12
        
        # 简单的径向函数近似 (Si 3s/3p)
        # STO: r^(n-1) * exp(-zeta * r)
        # n=3, zeta approx 1.38 (Slater rules for Si)
        zeta = 1.38
        radial = (r**2) * np.exp(-zeta * r)
        
        # 角向部分 (Spherical Harmonics / Real Orbitals)
        # s: 1
        # px: x/r
        # py: y/r
        # pz: z/r
        psi = np.zeros_like(r, dtype=complex)
        
        # s 轨道贡献
        c_s = coeffs[0]
        if abs(c_s) > 1e-6:
            psi += c_s * radial * 1.0
            
        # p 轨道贡献
        c_px, c_py, c_pz = coeffs[1], coeffs[2], coeffs[3]
        if abs(c_px) > 1e-6: psi += c_px * radial * (rx / r)
        if abs(c_py) > 1e-6: psi += c_py * radial * (ry / r)
        if abs(c_pz) > 1e-6: psi += c_pz * radial * (rz / r)
        
        return psi

    def plot_sp3_orbital(self, atom_idx: int = 0, hybrid_idx: int = 0, iso_level: float = 0.02):
        """
        绘制单个 sp3 杂化轨道的等值面。
        """
        # 1. 获取杂化系数
        hybrids = self.get_sp3_hybrids_at_atom(atom_idx)
        # 提取该原子上的 s, px, py, pz 系数
        n_orb = self.model.n_orb_atom
        base = atom_idx * n_orb
        coeffs = hybrids[hybrid_idx, base : base+4] # 只取 s,p 部分
        
        # 2. 定义空间网格
        L = 3.5 # 绘图范围 (Angstrom)
        N = 50
        x = np.linspace(-L, L, N)
        y = np.linspace(-L, L, N)
        z = np.linspace(-L, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 3. 计算波函数
        atom_pos = np.array([0,0,0]) # 相对原子位置
        psi = self.evaluate_orbital_spatial(coeffs, X, Y, Z, atom_pos)
        prob_density = np.abs(psi)**2
        
        # 4. 绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原子核
        ax.scatter([0], [0], [0], color='red', s=200, label='Atom')
        
        max_val = np.max(prob_density)
        mask = prob_density > max_val * iso_level
        # 使用 viridis colormap，高密度区颜色更亮
        sc = ax.scatter(X[mask], Y[mask], Z[mask], c=prob_density[mask], 
                   cmap='viridis', alpha=0.3, s=10, edgecolor='none')
        
        # 添加 colorbar
        plt.colorbar(sc, ax=ax, shrink=0.6, label='Probability Density')
        
        # 绘制指向方向
        direction = self.sp3_coeffs[hybrid_idx, 1:] # px, py, pz 分量即方向
        direction = direction / np.linalg.norm(direction) * 2.5
        ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], 
                  color='red', length=1.0, linewidth=2.0, label='Direction')

        ax.set_title(f'sp3 Hybrid Orbital #{hybrid_idx+1} on Atom {atom_idx}')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        plt.legend()
        plt.savefig('./pic/sp3.png',dpi=300)
        plt.show()

    def plot_bond_wannier(self):
        """
        绘制成键 Wannier 函数 (Bond Orbital)。
        对于 Si，这是两个相邻原子 sp3 轨道的线性组合：
        |W_bond> = 1/sqrt(2) (|h_A> + |h_B>)
        """
        # 取原子 A 的第 0 个 sp3 (指向 (1,1,1))
        # 对应的最近邻是 R = (a/4, a/4, a/4) 处的原子 B
        # 原子 B 指向 A 的 sp3 是 (-1,-1,-1) 方向的
        
        # A 的 sp3 (1,1,1)
        coeffs_A = self.sp3_coeffs[0] 
        # B 的 sp3 (-1,-1,-1) -> 对应索引 3: (1, -1, -1, 1) ? 需检查 sp3 定义
        # 让我们直接构造指向 (-1,-1,-1) 的 sp3
        coeffs_B_local = 0.5 * np.array([1, -1, -1, -1]) 
        
        # 网格
        a = self.model.params.a
        L = a * 0.7
        N = 60
        x = np.linspace(-L/2, L, N)
        y = np.linspace(-L/2, L, N)
        z = np.linspace(-L/2, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        pos_A = np.array([0, 0, 0])
        pos_B = np.array([a/4, a/4, a/4])
        
        psi_A = self.evaluate_orbital_spatial(coeffs_A, X, Y, Z, pos_A)
        psi_B = self.evaluate_orbital_spatial(coeffs_B_local, X, Y, Z, pos_B)
        
        # 成键态 (同相叠加)
        psi_bond = (psi_A + psi_B) / np.sqrt(2)
        prob = np.abs(psi_bond)**2
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        center = (pos_A + pos_B)/2
        ax.scatter([center[0]], [center[1]], [center[2]], color='red', marker='*', s=200, label='Bond Center')
        
        # 原子
        ax.scatter([pos_A[0], pos_B[0]], [pos_A[1], pos_B[1]], [pos_A[2], pos_B[2]], 
                   color='red', s=100)
        ax.text(pos_A[0], pos_A[1], pos_A[2]-0.2, "Si A")
        ax.text(pos_B[0], pos_B[1], pos_B[2]+0.2, "Si B")
        
        max_prob = np.max(prob)
        iso = max_prob * 0.25 # 提高阈值，只显示最强的 75% 区域
        mask = prob > iso
        
        # 使用 'plasma' colormap (火热的感觉)，增加 alpha 和 s
        sc = ax.scatter(X[mask], Y[mask], Z[mask], 
                   c=prob[mask], cmap='plasma', alpha=0.4, s=15, edgecolor='none')
        
        plt.colorbar(sc, ax=ax, shrink=0.6, label='Charge Density |Ψ|²')
        
        ax.set_title('Si-Si Bond Wannier Function (sp3-sp3 σ bond)')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        
        # 设置视角以便更好地观察键
        ax.view_init(elev=30, azim=45)
        
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # 1. 初始化模型
    params = SKParameters()
    model = SKTightBinding(params, basis="sp3") # 使用 sp3 基组演示更清晰
    
    # 2. 初始化 Wannier 本地化器
    wannier = WannierLocalizer(model)
    
    # 3. 绘制单个 sp3 杂化轨道
    print("Plotting single sp3 hybrid orbital...")
    wannier.plot_sp3_orbital(atom_idx=0, hybrid_idx=0)
    
    # 4. 绘制成键 Wannier 函数 (Bond Orbital)
    print("Plotting Si-Si bond Wannier function...")
    wannier.plot_bond_wannier()
    
    # 5. 计算 Wannier 中心
    centers = wannier.compute_wannier_centers()
    print("\nCalculated Wannier Centers (Bond Centers):")
    for i, c in enumerate(centers):
        print(f"Bond {i+1}: {c} Å")