import numpy as np
import matplotlib.pyplot as plt

# 从qq.csv文件加载电荷量数据，第二列是电荷量，存入charges数组
charges = np.loadtxt("qq.csv" , delimiter=',', usecols=1)  # 跳过第一行标题，读取第二列数据

# 调整参数
expected_e = 1.6e-19  # 预期基本电荷量（用于指导参数设置）
dt = expected_e / 50   # 时间分辨率设置为预期e的1/20
t_min = 0              # 数据最小值为1.6e-19，可从0开始
t_max = 1.7e-18        # 扩展时间轴至覆盖最大电荷值

# 生成时间轴和冲激信号
num_bins = int((t_max - t_min) / dt) + 1
time_axis = np.linspace(t_min, t_max, num_bins)
signal = np.zeros_like(time_axis)

# 精确匹配电荷位置到时间轴索引
indices = ((charges - t_min) / dt).astype(int)
valid_indices = indices[(indices >= 0) & (indices < len(signal))]
for idx in valid_indices:
    signal[idx] += 1

# 傅里叶变换
fft_result = np.fft.fft(signal+signal)  # 计算FFT，信号长度加倍以避免频谱混叠
freq = np.fft.fftfreq(2*len(signal), d=dt)
amplitude = np.abs(fft_result)

# 提取正频率并找到主峰
positive_mask = (freq > 0) & (freq < 1.8e19)  # 限制频率范围
freq_positive = freq[positive_mask]
amp_positive = amplitude[positive_mask]
main_freq = freq_positive[np.argmax(amp_positive[1:]) + 1]  # 跳过直流分量
e_estimated = 1 / main_freq

# 可视化
plt.figure(figsize=(16, 6))

# 时域信号
plt.subplot(2, 1, 1)
plt.stem(time_axis, signal, markerfmt=' ', basefmt=' ')
plt.title('Charge Signal in Time Domain')
plt.xlabel(r'$q/C$')
plt.ylabel('Signal Amplitude')

# 频域
plt.subplot(2, 1, 2)
plt.plot(freq_positive, amp_positive)
plt.axvline(main_freq, color='r', linestyle='--', 
            label=f'main frequency: {main_freq:.3e} 1/C \n estimated_e: {e_estimated:.3e} C')
plt.xlim(0, 1.8e19)  # 聚焦合理频率范围
plt.xlabel('frequency (1/C)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
#plt.show()
#plt.savefig('fft_analysis.png', dpi=300)  # 保存图像