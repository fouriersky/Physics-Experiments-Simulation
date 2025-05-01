import numpy as np
import matplotlib.pyplot as plt

class Brown:
    def __init__(self, r=0.5e-6, eta=1.8e-5, T=300, dt=0.1, total_time=120, noise_std=0.1e-6):
        self.r = r
        self.eta = eta
        self.T = T
        self.k = 1.38e-23
        self.dt = dt
        self.total_time = total_time
        self.noise_std = noise_std

        self.num_steps = int(total_time / dt)
        self.time = np.arange(0, total_time, dt)
        self.lambda_air = 0.1e-6  # 平均自由程
        self.D_theory = (self.k * T) / (6 * np.pi * eta * r) * (1 + self.lambda_air / r)

        self.x = np.zeros(self.num_steps)
        self.y = np.zeros(self.num_steps)
        self._simulate_trajectory()
        self._add_drift_noise()
        self._compute_msd_md()

    def _simulate_trajectory(self):
        step_std = np.sqrt(2 * self.D_theory * self.dt)
        for i in range(1, self.num_steps):
            dx = np.random.normal(0, step_std)
            dy = np.random.normal(0, step_std)
            seed_y = np.random.uniform(0, 0.2)
            self.x[i] = self.x[i-1] + dx
            self.y[i] = self.y[i-1] + dy * (seed_y + 1)

    def _add_drift_noise(self):
        drift_x = np.linspace(0, 1e-6, self.num_steps) + 1e-7 * np.sin(0.01 * np.arange(self.num_steps))
        drift_y = np.linspace(0, -1e-6, self.num_steps) + 1e-7 * np.cos(0.01 * np.arange(self.num_steps))
        self.x += drift_x + np.random.normal(0, self.noise_std, self.num_steps)
        self.y += drift_y + np.random.normal(0, self.noise_std, self.num_steps) + self.num_steps * 0.01
        self.Radius = np.sqrt(self.x**2 + self.y**2) + np.random.normal(0, self.noise_std, self.num_steps)

    def compute_msd(self, trajectory):
        max_lag = 120
        lags = np.arange(1, max_lag)
        msd = [np.mean((trajectory[lag:] - trajectory[:-lag]) ** 2) for lag in lags]
        return lags * self.dt, np.array(msd)

    def compute_md(self, trajectory):
        max_lag = 120
        lags = np.arange(1, max_lag)
        md = [np.mean((trajectory[lag:] - trajectory[:-lag])) for lag in lags]
        return lags * self.dt, np.array(md)

    def _compute_msd_md(self):
        self.time_lags, self.msd_r = self.compute_msd(self.Radius)

        # 添加后期扰动和平滑过渡
        noise_trend = np.zeros_like(self.time_lags)
        wave_amplitude = 1e-12
        wave_frequency = 1.2
        trend_strength = 5e-13

        start_idx = int(len(self.time_lags) * 0.4)
        end_idx = len(self.time_lags)
        transition_weights = 0.5 * (1 - np.cos(np.linspace(0, np.pi, end_idx - start_idx)))

        for j, i in enumerate(range(start_idx, end_idx)):
            t = self.time_lags[i]
            weight = transition_weights[j]
            wave = wave_amplitude * np.sin(wave_frequency * t)
            trend = trend_strength * t * 0.1
            noise_trend[i] = weight * (wave + trend)

        self.msd_r_plot = self.msd_r + noise_trend

        # 添加平均位移扰动
        t_index = np.arange(self.num_steps)
        amp_x, amp_y = 3e-5, 3e-5
        freq_x, freq_y = 0.04, 0.05
        self.plot_x = self.x + amp_x * np.sin(freq_x * t_index) + np.random.normal(0, 0.01e-6, self.num_steps)
        self.plot_y = self.y + amp_y * np.cos(freq_y * t_index) + np.random.normal(0, 0.01e-6, self.num_steps)
        _, self.mean_disp_x = self.compute_md(self.plot_x)
        _, self.mean_disp_y = self.compute_md(self.plot_y)

        # 拟合D
        coeff = np.polyfit(self.time_lags, self.msd_r, 1)
        self.D_simulated = coeff[0] / 2

    def report_diffusion(self):
        print(f"理论扩散系数 D = {self.D_theory:.2e} m²/s")
        print(f"模拟扩散系数 D = {self.D_simulated:.2e} m²/s")

    def plot_all(self):
        plt.figure(figsize=(16, 4))

        # 轨迹图
        plt.subplot(1, 3, 1)
        plt.plot(self.x * 1e6, self.y * 1e6, alpha=0.6)
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.title("Brownian Motion Trajectory")
        plt.gca().get_yaxis().get_offset_text().set_visible(False)

        # MSD图
        plt.subplot(1, 3, 2)
        plt.plot(self.time_lags * 100, self.msd_r_plot * 1e12, label='sim MSD', linewidth=2)
        plt.plot(self.time_lags * 100, 2 * self.D_theory * self.time_lags * 1e12, '--', label='theory MSD', color='red')
        plt.xlabel("t (s)")
        plt.ylabel("MSD (µm²)")
        plt.title("MSD vs. Time")
        plt.legend()

        # MD图
        plt.subplot(1, 3, 3)
        plt.plot(self.time_lags * 100, self.mean_disp_x * 1e6, label='X')
        plt.plot(self.time_lags * 100, self.mean_disp_y * 1e6, label='Y')
        plt.xlabel("t (s)")
        plt.ylabel("Drift (µm)")
        plt.title("Mean Drift of X/Y")
        plt.legend()

        plt.tight_layout()
        plt.show()
