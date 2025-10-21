import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 参数（无量纲）
alpha = 15.6
beta = 25.6
m0 = -8/7
m1 = -5/7

def f(x):
    return m1*x + 0.5*(m0 - m1)*(np.abs(x+1) - np.abs(x-1))

# 蔡氏电路微分方程
def chua(t, state):
    x, y, z = state
    dx = alpha * (y - x - f(x))
    dy = x - y + z
    dz = -beta * y
    return [dx, dy, dz]

# 初始条件
state0 = [0.1, 0.0, 0.0]
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 20000)

# 数值积分
sol = solve_ivp(chua, t_span, state0, t_eval=t_eval)

x, y, z = sol.y

# 绘制相图 (x, y)
plt.figure(figsize=(6,6))
plt.plot(x, y, lw=0.3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Chua's Attractor")
plt.grid(True)
#plt.tight_layout()
plt.savefig("./experiments/chua.png",dpi=300)
