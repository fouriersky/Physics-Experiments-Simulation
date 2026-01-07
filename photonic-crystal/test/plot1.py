import numpy as np
import matplotlib.pyplot as plt

# 配置：填写你的 CSV 路径
CSV_PATH = r"e:\python_code\Physics-Experiments-Simulation\photonic-crystal\test\221band.csv"
OUT_PNG  = r"e:\python_code\Physics-Experiments-Simulation\photonic-crystal\test\221band1.png"

G = np.array([0.0,  0.0,  0.0])    # Γ
X = np.array([0.0,  0.5,  0.5])    # X（面心）
W = np.array([0.25, 0.75, 0.5])    # W（棱中点）
L = np.array([0.5,  0.5,  0.5])    # L（角点）

def load_csv(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    k_dist = data[:, 0]
    freqs = data[:, 1:]
    return k_dist, freqs

def segment_cumulative_positions_by_coords(total_length):
    """
    根据几何坐标计算各段长度比例，得到累计距离上的刻度位置：
    0, |GX|, |GX|+|XR|, |GX|+|XR|+|RM|, |GX|+|XR|+|RM|+|MG|
    并按总长度进行缩放到 k_dist 轴。
    """
    L_GX = np.linalg.norm(X - G)
    L_XR = np.linalg.norm(W - X)
    L_RM = np.linalg.norm(L - W)
    L_MG = np.linalg.norm(G - L)
    L_total_geom = L_GX + L_XR + L_RM + L_MG
    scale = total_length / L_total_geom if L_total_geom > 0 else 1.0
    cum = np.array([0.0, L_GX, L_GX + L_XR, L_GX + L_XR + L_RM, L_total_geom], dtype=float)
    return cum * scale

def nearest_indices(k_dist, xs):
    idxs = []
    for x in xs:
        idxs.append(int(np.argmin(np.abs(k_dist - x))))
    return idxs

def plot_bands(csv_path, out_png):
    k_dist, freqs = load_csv(csv_path)
    total_len = k_dist[-1]
    # 用坐标几何长度确定刻度位置
    xs = segment_cumulative_positions_by_coords(total_len)
    labels = [r"$\Gamma$", "X", "W", "L", r"$\Gamma$"]
    # 找到最接近的索引（用于画竖线更贴合采样点）
    idxs = nearest_indices(k_dist, xs)
    xs_snap = [k_dist[i] for i in idxs]

    plt.figure(figsize=(8, 8))
    # 绘制能带
    for i in range(freqs.shape[1]):
        plt.plot(k_dist, freqs[:, i], lw=1.3, color="C0")

    # 分段竖线与刻度
    for x in xs_snap:
        plt.axvline(x, color="k", lw=0.7, alpha=0.35)
    plt.xticks(xs, labels)
    plt.ylabel("Frequency (2πc/a)")
    plt.title("Bands: Γ-X-W-L-Γ (fcc)")
    plt.grid(alpha=0.15)
    plt.ylim(0,0.7)
    plt.xlim(k_dist[0],k_dist[-1])
    plt.tight_layout()
    plt.show()
    plt.savefig(out_png, dpi=300)

if __name__ == "__main__":
    plot_bands(CSV_PATH, OUT_PNG)