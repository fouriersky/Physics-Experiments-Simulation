import numpy as np
import matplotlib.pyplot as plt


class IsingModel:
    def __init__(self,L=None, J=None, H=None ,steps=None, temperature=None):
        self.L = L  
        self.J = J   
        self.H = H
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
                energy += -self.J * S * neighbors - self.H * S
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
        neighbor_sum = left + right + up + down
        return 2 * self.spins[i, j] * (self.J * neighbor_sum + self.H)


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
        avg_energies = []

        for T in self.temperature:
            self.spins = self.initialize_spins()
            energies = []

            # 模拟步数
            for step in range(self.steps):
                self.metropolis(T)
                if step > self.steps // 2: 
                    energies.append(self.calc_total_energy())

            # 计算平均能量
            avg_energy = np.mean(energies)
            avg_energies.append(avg_energy)

        # 计算比热
        for i in range(1, len(self.temperature)):
            dE = avg_energies[i] - avg_energies[i - 1]
            dT = self.temperature[i] - self.temperature[i - 1]
            specific_heat = dE / dT / (self.L**2)  # 归一化到每个格点
            specific_heats.append(specific_heat)

        return specific_heats, avg_energies[1:]

    def plot_results(self, specific_heats):
        plt.figure(figsize=(6, 5))
        plt.plot(self.temperature[1:], specific_heats,  label=r'$C_v$')
        plt.axvline(x=2.27, color='r', linestyle='--', label=r'$T_c$')
        plt.xlabel('T')
        plt.ylabel(r'$C_v$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ising_model_cv.png', dpi=300)

if __name__ == "__main__":
    model = IsingModel(
        L=50, 
        J=1, 
        H=0,
        steps=200, 
        temperature=np.linspace(2.0, 3.0, 100)
    )
    specific_heats , avg_energies = model.simulate()
    model.plot_results(specific_heats)