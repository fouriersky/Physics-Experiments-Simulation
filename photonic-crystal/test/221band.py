import os
import numpy as np
import meep as mp
from meep import mpb

# 简单立方晶格与几何（与 irrep225.py 一致）
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1, 1),
    basis1=mp.Vector3(1, 0, 0),
    basis2=mp.Vector3(0, 1, 0),
    basis3=mp.Vector3(0, 0, 1)
)

eps_sphere = 13.0
radius = 0.30
diel = mp.Medium(epsilon=eps_sphere)
geometry = [mp.Sphere(radius, center=mp.Vector3(0, 0, 0), material=diel)]

# 计算设置
resolution = 32        # 可根据需要提高
num_bands = 12         # 计算的能带数量
nseg = 20              # 每段的插值点数（包含端点），总点数约 4*(nseg) - 3

# 路径：Γ → X → R → M → Γ
G = mp.Vector3(0.0, 0.0, 0.0)
X = mp.Vector3(0.0, 0.0, 0.5)       # 约定 X=(0,0,1/2)
R = mp.Vector3(0.5, 0.5, 0.5)
M = mp.Vector3(0.5, 0.5, 0.0)

# 拼接各段 k 点，并插值
k_path = []
k_path += mp.interpolate(nseg, [G, X])
k_path += mp.interpolate(nseg, [X, R])[1:]  # 去掉重复端点
k_path += mp.interpolate(nseg, [R, M])[1:]
k_path += mp.interpolate(nseg, [M, G])[1:]

# 运行 MPB 只求频率
ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_path,
    resolution=resolution,
    num_bands=num_bands
)
ms.run()  

# 提取并保存结果
# ms.all_freqs: 形状 (nk, num_bands)
freqs = np.array(ms.all_freqs)
dist = [0.0]
for i in range(1, len(k_path)):
    dk = np.array([k_path[i].x - k_path[i-1].x,
                   k_path[i].y - k_path[i-1].y,
                   k_path[i].z - k_path[i-1].z], dtype=float)
    dist.append(dist[-1] + np.linalg.norm(dk))
dist = np.array(dist)

# 保存 CSV：第一列为累计距离，其余列为各带频率
out_csv = "bands_sc_G-X-R-M-G.csv"
data_to_save = np.column_stack([dist, freqs])
np.savetxt(out_csv, data_to_save, delimiter=",", header="k_dist," + ",".join([f"band_{i+1}" for i in range(num_bands)]), comments="")

# 控制台简要输出
print(f"完成路径 Γ→X→R→M→Γ 的频率计算")
print(f"点数: {len(k_path)}, 频带数: {num_bands}, CSV: {out_csv}")
print("首个 k 点（Γ）频率前几条：", freqs[0, :min(6, num_bands)])