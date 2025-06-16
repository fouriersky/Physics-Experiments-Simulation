import numpy as np
import matplotlib.pyplot as plt


class IsingModel:
    def __init__(self,L=None, J=None, steps=None, temperature=None):
        self.L = L  
        self.J = J   
        self.steps = steps  
        self.temperature = temperature
        self.spins = self.initialize_spins()

    # 初始化自旋格子
    def initialize_spins(self):
        return np.random.choice([-1, 1], size=(self.L, self.L))

    def calc_total_energy(self):
        """计算系统的总能量"""
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                S = self.spins[i, j]
                neighbors = (
                    self.spins[(i + 1) % self.L, j]
                    + self.spins[i, (j + 1) % self.L]
                    + self.spins[(i - 1) % self.L, j]
                    + self.spins[i, (j - 1) % self.L]
                )
                energy += -self.J * S * neighbors
        return energy / 2  

    def calc_magnetization(self):
        """计算系统的总磁化强度"""
        return np.sum(self.spins)

    def calc_energy_change(self, i, j):
        """计算(i,j)点的能量变化"""
        left = self.spins[i, (j - 1) % self.L]
        right = self.spins[i, (j + 1) % self.L]
        up = self.spins[(i - 1) % self.L, j]
        down = self.spins[(i + 1) % self.L, j]
        return 2 * self.J * self.spins[i, j] * (left + right + up + down)


    def metropolis(self, T):
        """Metropolis算法更新一次"""
        for _ in range(self.L * self.L):
            i = np.random.randint(0, self.L)
            j = np.random.randint(0, self.L)
            dE = self.calc_energy_change(i, j)
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                self.spins[i, j] *= -1

    def simulate(self):
        """模拟不同温度下的物理量"""
        specific_heats = []
        susceptibilities = []

        for T in self.temperature:
            self.spins = self.initialize_spins()
            energies = []
            magnetizations = []

            # 模拟步数
            for step in range(self.steps):
                self.metropolis(T)
                if step > self.steps // 2:  # 热平衡后开始采样
                    energies.append(self.calc_total_energy())
                    magnetizations.append(self.calc_magnetization())

            # 计算物理量
            avg_energy = np.mean(energies)
            avg_energy_sq = np.mean(np.square(energies))
            avg_magnetization = np.mean(magnetizations)
            avg_magnetization_sq = np.mean(np.square(magnetizations))

            specific_heat = (avg_energy_sq - avg_energy**2) / (T**2 * self.L**2)
            susceptibility = (avg_magnetization_sq - avg_magnetization**2) / (T * self.L**2)

            specific_heats.append(specific_heat)
            susceptibilities.append(susceptibility)

        return specific_heats, susceptibilities

    def plot_results(self, specific_heats, susceptibilities):
        """绘制比热和磁化率随温度的变化"""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.temperature, specific_heats, 'o-', label=r'$C_v$')
        plt.axvline(x=2.269, color='r', linestyle='--')
        plt.xlabel('T')
        plt.ylabel(r'$C_v$')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.temperature, susceptibilities, 'o-', label=r'$\chi$')
        plt.axvline(x=2.269, color='r', linestyle='--', label=r'$T_c$')
        plt.xlabel('T')
        plt.ylabel(r'$\chi$')
        plt.legend()

        plt.tight_layout()
        plt.savefig('ising_model_results.png',dpi=300)

if __name__ == "__main__":
    model = IsingModel(
        L=50, 
        J=1, 
        steps=800, 
        temperature=np.linspace(2.0, 3.0, 30)
    )
    specific_heats, susceptibilities = model.simulate()
    model.plot_results(specific_heats, susceptibilities)