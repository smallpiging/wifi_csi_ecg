import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取数据集路径
check_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pca_datasets')
csv_list = [f for f in os.listdir(check_path) if os.path.splitext(f)[-1] == '.csv']
print(f"📁 找到的 CSV 文件: {csv_list}")

csv_path = csv_list[0]

df = pd.read_csv(os.path.join(check_path, csv_path))

# 智能选择列名：优先看提纯后的 PC1，如果没有就看原始的 CSI_Mag_1
target_col = 'PC1' if 'PC1' in df.columns else 'CSI_Mag_3'

if target_col in df.columns:
    print(f"📊 正在分析列: {target_col}")
    signal = df[target_col].values
    label = df['ECG_Heatmap_Label'].values

    fs = 125  # 采样率
    N = len(signal)

    # ==========================================
    # 核心 1：计算 WiFi CSI 频谱
    # ==========================================
    signal_centered = signal - np.mean(signal)
    fft_signal = np.fft.fft(signal_centered)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    half_n = N // 2
    pos_freqs = freqs[:half_n]
    pos_fft_mag_signal = np.abs(fft_signal[:half_n]) * 2 / N

    # ==========================================
    # 核心 2：计算 ECG 真实标签频谱 (你的新需求)
    # ==========================================
    label_centered = label - np.mean(label)
    fft_label = np.fft.fft(label_centered)
    pos_fft_mag_label = np.abs(fft_label[:half_n]) * 2 / N

    # ==========================================
    # 核心 3：自动化心率提取与误差对比
    # ==========================================
    # 锁定 0.8 Hz 到 2.0 Hz 的心跳红区
    heart_band_indices = np.where((pos_freqs >= 0.8) & (pos_freqs <= 2.0))[0]
    heart_freqs = pos_freqs[heart_band_indices]

    # 提取 CSI (预测) 最高峰
    mags_signal = pos_fft_mag_signal[heart_band_indices]
    peak_idx_signal = np.argmax(mags_signal)
    pred_freq = heart_freqs[peak_idx_signal]
    pred_hr = pred_freq * 60
    pred_mag = pos_fft_mag_signal[heart_band_indices][peak_idx_signal]

    # 提取 ECG (真实) 最高峰
    mags_label = pos_fft_mag_label[heart_band_indices]
    peak_idx_label = np.argmax(mags_label)
    true_freq = heart_freqs[peak_idx_label]
    true_hr = true_freq * 60
    true_mag = pos_fft_mag_label[heart_band_indices][peak_idx_label]

    print("-" * 50)
    print(f"🎯 频域对决 (Algorithm vs Ground Truth):")
    print(f"   ❤️  真实心率 (ECG): {true_hr:.1f} BPM ({true_freq:.3f} Hz)")
    print(f"   📡  预测心率 (CSI): {pred_hr:.1f} BPM ({pred_freq:.3f} Hz)")
    print(f"   ⚖️  绝对误差:       {abs(true_hr - pred_hr):.1f} BPM")
    print("-" * 50)

    # ==========================================
    # 画图展示 (4 行图表，极其硬核的对比)
    # ==========================================
    plt.figure(figsize=(16, 12))

    # --- 1. WiFi 时域 ---
    plt.subplot(4, 1, 1)
    plt.plot(signal, color='blue', linewidth=1)
    plt.title(f'[WIFI] Time Domain: {target_col}')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # --- 2. WiFi 频域 ---
    plt.subplot(4, 1, 2)
    plt.plot(pos_freqs, pos_fft_mag_signal, color='purple', linewidth=1.5)
    plt.plot(pred_freq, pred_mag, 'ro', markersize=8, label=f'Pred HR: {pred_hr:.1f} BPM')  # 打红点
    plt.title('[WIFI] Frequency Spectrum')
    plt.ylabel('Magnitude')
    plt.xlim(0, 5)
    plt.axvspan(0.8, 2.0, color='red', alpha=0.1, label='Heartbeat Range')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # --- 3. ECG 时域 ---
    plt.subplot(4, 1, 3)
    plt.plot(label, color='red', linewidth=1)
    plt.title('[ECG] Time Domain: Ground Truth Label')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)

    # --- 4. ECG 频域 ---
    plt.subplot(4, 1, 4)
    plt.plot(pos_freqs, pos_fft_mag_label, color='darkorange', linewidth=1.5)
    plt.plot(true_freq, true_mag, 'go', markersize=8, label=f'True HR: {true_hr:.1f} BPM')  # 打绿点
    plt.title('[ECG] Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 5)
    plt.axvspan(0.8, 2.0, color='red', alpha=0.1, label='Heartbeat Range')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print(f"❌ 找不到列 {target_col}，请检查你的 CSV 内容！")