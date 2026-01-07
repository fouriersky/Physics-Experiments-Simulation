import os
import numpy as np
import matplotlib.pyplot as plt

# 高对称点与标签（按路径顺序）
HS_POINTS = [
    ('X', (0.0, 0.5, 0.5)),
    ('U', (0.0, 0.625, 0.375)),
    ('L', (0.0, 0.5, 0.0)),
    ('Γ', (0.0, 0.0, 0.0)),
    ('X', (0.0, 0.5, 0.5)),
    ('W', (0.25, 0.75, 0.5)),
    ('K', (0.375, 0.75, 0.375)),
]

def parse_mpb_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # 头一行决定列含义与带的数量
    header = lines[0]
    header_cols = [c.strip() for c in header.split(',')]
    n_bands = sum(c.lower().startswith('band') for c in header_cols)

    # 数据行：freqs:, k_index, k1, k2, k3, kmag/2pi, band1..bandN
    k_list = []
    bands = []
    idx_list = []

    for ln in lines[1:]:
        # 只处理以 "freqs:" 开头的行
        if not ln.lower().startswith('freqs:'):
            continue
        cols = [c.strip() for c in ln.split(',')]
        # 容错：行长度不足则跳过
        if len(cols) < 6 + n_bands:
            continue

        try:
            k_index = int(cols[1])
            k1 = float(cols[2]); k2 = float(cols[3]); k3 = float(cols[4])
            # cols[5] 是 kmag/2pi，此处不直接使用
            band_vals = [float(x) for x in cols[6:6 + n_bands]]
        except ValueError:
            # 有异常就跳过这行
            continue

        idx_list.append(k_index)
        k_list.append([k1, k2, k3])
        bands.append(band_vals)

    k = np.array(k_list, dtype=float)            # (N, 3)
    freqs = np.array(bands, dtype=float)         # (N, n_bands)
    idxs = np.array(idx_list, dtype=int)

    # 计算沿路径的累计长度（分段线性，单位是分数倒格矢坐标的欧氏距离）
    s = np.zeros(len(k))
    if len(k) > 1:
        steps = np.linalg.norm(np.diff(k, axis=0), axis=1)
        s[1:] = np.cumsum(steps)

    return s, k, freqs, idxs, n_bands

def find_breakpoints(s, k):
    # 用k点跳变（步长突增）自动检测段落分界
    if len(s) < 2:
        return [0]
    steps = np.diff(s)
    # 正常步长的典型值
    finite_steps = steps[steps > 0]
    if len(finite_steps) == 0:
        return [0, len(s) - 1]
    typical = np.median(finite_steps)
    # 阈值：显著大于典型步长视为段落切换
    thresh = 3.0 * typical
    breaks = [i for i, ds in enumerate(steps, start=1) if ds > thresh]

    # 段落起止下标
    ticks = [0] + breaks + [len(s) - 1]
    # 去重并排序
    ticks = sorted(set(ticks))
    return ticks

def k_label(vec):
    # Γ 点识别
    if np.linalg.norm(vec) < 1e-9:
        return 'Γ'
    # 其他点显示坐标（尽量紧凑）
    def fmt(x):
        # 规整到最多3位小数，去掉多余0
        return ('{:g}'.format(float(f'{x:.3f}')))
    return f'({fmt(vec[0])},{fmt(vec[1])},{fmt(vec[2])})'

def find_ticks_from_hs_points(s, k, hs_points, atol=1e-6):
    # 根据给定高对称点坐标顺序，定位对应的索引并生成刻度
    xticks = []
    xlabels = []
    found = 0
    for label, vec in hs_points:
        vec = np.asarray(vec, dtype=float)
        # 找到与该vec最近的k点索引
        d = np.linalg.norm(k - vec, axis=1)
        i = int(np.argmin(d)) if len(d) else 0
        if len(d) and d[i] <= atol:
            xticks.append(s[i])
            xlabels.append(label)
            found += 1
        else:
            # 若没在容差内命中，则依旧用最近点，但标注在该处
            if len(d):
                xticks.append(s[i])
                xlabels.append(label)
            # 若没有数据点，跳过
    return xticks, xlabels, found

def plot_bands(s, k, freqs, save_png='band_structure.png', save_csv='bands_extracted.csv', hs_points=None):
    n_bands = freqs.shape[1]

    xticks = None
    xlabels = None
    if hs_points:
        xticks, xlabels, found = find_ticks_from_hs_points(s, k, hs_points, atol=1e-6)
        # 如果匹配太少，退回自动分段
        if found < 2:
            ticks = find_breakpoints(s, k)
            xticks = [s[i] for i in ticks]
            xlabels = [k_label(k[i]) for i in ticks]
    else:
        ticks = find_breakpoints(s, k)
        xticks = [s[i] for i in ticks]
        xlabels = [k_label(k[i]) for i in ticks]

    plt.figure(figsize=(7.5, 5))
    for b in range(n_bands):
        plt.plot(s, freqs[:, b], lw=1.5, label=f'Band {b+1}')
    for x in xticks:
        plt.axvline(x, color='k', lw=0.8, alpha=0.5)

    plt.xlim(s[0], s[-1] if len(s) else 1.0)
    plt.ylim(0,None)
    plt.xlabel('')
    plt.ylabel('Frequency')
    plt.title('Band Structure Fm3m(225) 8 bands')
    plt.xticks(xticks, xlabels, rotation=0)
    plt.grid(True, which='both', ls=':', alpha=0.4)
    # plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_png, dpi=180)

    # 导出数据为 CSV：第一列是路径坐标 s，其后是各条带
    #out = np.column_stack([s, freqs])
    #header = 's,' + ','.join([f'band_{i+1}' for i in range(n_bands)])
    #np.savetxt(save_csv, out, delimiter=',', header=header, comments='', fmt='%.6f')
    print(f'save: {save_png}')
    #print(f'save: {save_csv}')

def main():
    # 输入文件路径
    root = os.path.dirname(__file__)
    txt_path = os.path.join(root, 'data225.txt')

    s, k, freqs, idxs, n_bands = parse_mpb_txt(txt_path)
    if len(s) == 0:
        print('未解析到有效数据。')
        return
    plot_bands(s, k, freqs,
               save_png=os.path.join(root, 'band_structure_225.png'),
               #save_csv=os.path.join(root, 'bands_extracted.csv'),
               hs_points=HS_POINTS)

if __name__ == '__main__':
    main()