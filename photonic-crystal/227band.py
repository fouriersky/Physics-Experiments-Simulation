import os
import math
import numpy as np
import meep as mp
from meep import mpb

# ---------- FCC primitive lattice ----------
# 与 irrep227.py 一致：a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)，|ai|=1/√2
geometry_lattice = mp.Lattice(
    basis_size=mp.Vector3(math.sqrt(0.5), math.sqrt(0.5), math.sqrt(0.5)),
    basis1=mp.Vector3(0, 1, 1),
    basis2=mp.Vector3(1, 0, 1),
    basis3=mp.Vector3(1, 1, 0)
)

# ---------- geometry: diamond (Fd-3m) 两球基元 ----------
eps = 13.0
r   = 0.25
diel = mp.Medium(epsilon=eps)
geometry = [
    mp.Sphere(r, center=mp.Vector3( 0.125,  0.125,  0.125), material=diel),
    mp.Sphere(r, center=mp.Vector3(-0.125, -0.125, -0.125), material=diel),
]

# ---------- 计算参数 ----------
resolution = 32       # 可提高获得更平滑的频率
num_bands  = 12       # 计算的带数
nseg       = 20       # 每段插值点数（包含端点），总点数约 4*nseg - 3

# ---------- 高对称 k 点（相对原胞反基坐标） ----------
G = mp.Vector3(0.0,  0.0,  0.0)    # Γ
X = mp.Vector3(0.0,  0.5,  0.5)    # X（面心）
W = mp.Vector3(0.25, 0.75, 0.5)    # W（棱中点）
L = mp.Vector3(0.5,  0.5,  0.5)    # L（角点）

# ---------- 路径：Γ → X → W → L → Γ ----------
k_path = []
k_path += mp.interpolate(nseg, [G, X])
k_path += mp.interpolate(nseg, [X, W])[1:]  # 去重复端点
k_path += mp.interpolate(nseg, [W, L])[1:]
k_path += mp.interpolate(nseg, [L, G])[1:]

# ---------- 运行 MPB（仅频率） ----------
ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_path,
    resolution=resolution,
    num_bands=num_bands
)
ms.filename_prefix = "bands_fcc_G-X-W-L-G"
ms.run()  # 不输出场

# ---------- 提取并保存 ----------
freqs = np.array(ms.all_freqs)  # 形状 (nk, num_bands)

# 按 k 点累计距离（使用分数坐标差的欧氏长度）
dist = [0.0]
for i in range(1, len(k_path)):
    dk = np.array([k_path[i].x - k_path[i-1].x,
                   k_path[i].y - k_path[i-1].y,
                   k_path[i].z - k_path[i-1].z], dtype=float)
    dist.append(dist[-1] + np.linalg.norm(dk))
dist = np.array(dist)

out_csv = "bands_fcc_G-X-W-L-G.csv"
data = np.column_stack([dist, freqs])
np.savetxt(out_csv, data, delimiter=",",
           header="k_dist," + ",".join([f"band_{i+1}" for i in range(num_bands)]),
           comments="")
