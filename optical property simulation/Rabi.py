import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def simulate_rabi_oscillation(Omega, Delta, t_max, steps=500):
    """
    模拟二能级系统的 Rabi 振荡
    
    参数:
    Omega : float -> 拉比频率 (Rabi frequency)，代表耦合强度
    Delta : float -> 失谐量 (Detuning), Delta = omega_drive - omega_0
    t_max : float -> 模拟的总时间
    steps : int   -> 时间步数
    
    返回:
    times : array -> 时间轴
    P_e   : array -> 激发态布居数概率
    """
    
    # 1. 定义哈密顿量 (在旋转波近似 RWA 下)
    # H = (hbar/2) * [ -Delta   Omega ]
    #                [  Omega   Delta ]
    # 这里假设 hbar = 1
    H = 0.5 * np.array([
        [-Delta, Omega],
        [Omega,  Delta]
    ])
    
    # 2. 定义初始状态
    # 我们假设从基态开始
    # |e> = [1, 0] (激发态)
    # |g> = [0, 1] (基态)
    psi_0 = np.array([0, 1], dtype=complex) 
    
    times = np.linspace(0, t_max, steps)
    P_e = [] # 记录激发态概率
    
    # 3. 时间演化计算
    # |psi(t)> = exp(-iHt) |psi(0)>
    for t in times:
        # 计算时间演化算符 U(t)
        U_t = expm(-1j * H * t)
        
        # 作用在初始态上
        psi_t = np.dot(U_t, psi_0)
        
        # 计算激发态 |e> 的概率 (取复数模的平方)
        # psi_t[0] 对应激发态分量
        prob_excited = np.abs(psi_t[0])**2
        P_e.append(prob_excited)
        
    return times, np.array(P_e)

# ==========================================
# 配置参数区域
# ==========================================

# 情况 1: 完全共振 (Resonance)
Omega_res = 2.0 * np.pi * 1.0  # 拉比频率 = 1 Hz (归一化单位)
Delta_res = 0.0                # 失谐 = 0

# 情况 2: 存在失谐 (Detuned)
Omega_det = 2.0 * np.pi * 1.0
Delta_det = 2.0 * np.pi * 2.0  # 失谐较大

# 模拟时间
t_end = 3.0 # 模拟 3 秒

# ==========================================
# 运行模拟
# ==========================================
t1, pe1 = simulate_rabi_oscillation(Omega_res, Delta_res, t_end)
t2, pe2 = simulate_rabi_oscillation(Omega_det, Delta_det, t_end)

# ==========================================
# 绘图
# ==========================================
plt.figure(figsize=(10, 6))

# 绘制共振曲线
plt.plot(t1, pe1, label=r'Resonance ($\Delta=0, \Omega=2\pi$)', linewidth=2.5, color='blue')

# 绘制失谐曲线
plt.plot(t2, pe2, label=r'Detuned ($\Delta=4\pi, \Omega=2\pi$)', linewidth=2.5, color='red', linestyle='--')

# 理论上的广义拉比频率 (用于验证)
# Omega_eff = sqrt(Omega^2 + Delta^2)
# 周期 T = 2*pi / Omega_eff

plt.title('Rabi Oscillations: Two-Level System', fontsize=16)
plt.xlabel('Time ', fontsize=14)
plt.ylabel(r'Excited State Population $P_{|e\rangle}$', fontsize=14)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# 显示理论最大布居数 (对于失谐情况)
max_P_det = (Omega_det**2) / (Omega_det**2 + Delta_det**2)  
plt.axhline(max_P_det, color='red', linestyle=':', alpha=0.5, label='Max Theoretical (Detuned)')

plt.tight_layout()
plt.savefig('./pic/rabi.png',dpi=300) 
#plt.show()