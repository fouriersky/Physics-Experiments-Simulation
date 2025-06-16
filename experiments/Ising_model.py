import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


L = 50  
T = 3.0  
steps = 500  
J = 1  

# 初始化自旋格子
spins = np.random.choice([-1, 1], size=(L, L))

def calc_energy(spins, i, j):
    """计算(i,j)点的能量变化"""
    left = spins[i, (j-1)%L]
    right = spins[i, (j+1)%L]
    up = spins[(i-1)%L, j]
    down = spins[(i+1)%L, j]
    return 2 * J * spins[i, j] * (left + right + up + down)

def metropolis(spins, T):
    """Metropolis算法更新一次"""
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = calc_energy(spins, i, j)
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1
    return spins

# 动画帧数据
frames = []

for step in range(steps):
    spins = metropolis(spins, T)
    if step % 4 == 0:  # 每5步保存一帧
        frames.append(spins.copy())

# 绘制动画
fig = plt.figure(figsize=(5,5))
im = plt.imshow(frames[0], cmap='coolwarm', animated=True)

def update(frame):
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
ani.save('ising_model.gif', writer='pillow')
plt.close()