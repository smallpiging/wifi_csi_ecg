import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks


# ==========================================
# 新增：Hampel 滤波器核心算法
# ==========================================
def hampel_filter_pandas(input_array, window_size=20, n_sigmas=3):
    """
    使用 Pandas 实现的 Hampel 滤波器
    """
    series = pd.Series(input_array)
    rolling_median = series.rolling(window=2 * window_size + 1, center=True).median()
    MAD = 1.4826 * (series - rolling_median).abs().rolling(window=2 * window_size + 1, center=True).median()
    outlier_idx = (series - rolling_median).abs() > (n_sigmas * MAD)

    series_filtered = series.copy()
    series_filtered[outlier_idx] = rolling_median[outlier_idx]

    # 打印斩首数量
    num_outliers = outlier_idx.sum()
    print(f"🗡️ Hampel 扫描完毕: 发现并斩首了 {num_outliers} 个异常毛刺点！")

    return series_filtered.bfill().ffill().values


# 获取数据集路径
check_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_datasets')
csv_list = [f for f in os.listdir(check_path) if os.path.splitext(f)[-1] == '.csv']
print(f"📁 找到的 CSV 文件: {csv_list}")

# 你可以换不同的文件索引看看效果
csv_path = csv_list[6]
df = pd.read_csv(os.path.join(check_path, csv_path))

target_col = 'PC1' if 'PC1' in df.columns else 'CSI_Mag_5'

if target_col in df.columns:
    print(f"📊 正在分析列: {target_col}")
    signal_raw = df[target_col].values
    label = df['Aligned_ECG'].values
    fs = 125
    N = len(signal_raw)

    # ==========================================
    # 核心 0：执行 Hampel 滤波
    # ==========================================
    # window_size=20 意味着前后各看20个点 (总窗口约0.3秒)
    signal_hampel = hampel_filter_pandas(signal_raw, window_size=50, n_sigmas=3)


    # ==========================================
    # 辅助函数：统一处理 FFT 和找峰逻辑 (避免代码重复)
    # ==========================================
    def get_fft_and_peak(sig):
        sig_centered = sig - np.mean(sig)
        fft_y = np.fft.fft(sig_centered)
        freqs = np.fft.fftfreq(N, d=1 / fs)

        half_n = N // 2
        pos_freqs = freqs[:half_n]
        pos_fft_mag = np.abs(fft_y[:half_n]) * 2 / N

        heart_band_indices = np.where((pos_freqs >= 0.8) & (pos_freqs <= 2.0))[0]
        heart_freqs = pos_freqs[heart_band_indices]
        mags = pos_fft_mag[heart_band_indices]

        # 找峰
        many_peaks, _ = find_peaks(mags, distance=2, prominence=0.015)
        if many_peaks.size > 0:
            peak_idx = many_peaks[np.argmax(mags[many_peaks])]
        else:
            peak_idx = np.argmax(mags)

        p_freq = heart_freqs[peak_idx]
        p_mag = mags[peak_idx]
        p_hr = p_freq * 60
        return pos_freqs, pos_fft_mag, p_freq, p_mag, p_hr


    # ==========================================
    # 分别计算三种信号的频谱和心率
    # ==========================================
    # 1. 原始信号
    freqs_raw, mag_raw, peak_f_raw, peak_m_raw, hr_raw = get_fft_and_peak(signal_raw)

    # 2. Hampel 滤波后信号
    freqs_hmp, mag_hmp, peak_f_hmp, peak_m_hmp, hr_hmp = get_fft_and_peak(signal_hampel)

    # 3. ECG 真实标签 (ECG 直接找最大值，不需要 find_peaks)
    label_centered = label - np.mean(label)
    fft_label = np.fft.fft(label_centered)
    freqs_ecg = np.fft.fftfreq(N, d=1 / fs)[:N // 2]
    mag_ecg = (np.abs(fft_label[:N // 2]) * 2 / N)

    hb_idx_ecg = np.where((freqs_ecg >= 0.8) & (freqs_ecg <= 2.0))[0]
    peak_idx_ecg = np.argmax(mag_ecg[hb_idx_ecg])
    peak_f_ecg = freqs_ecg[hb_idx_ecg][peak_idx_ecg]
    peak_m_ecg = mag_ecg[hb_idx_ecg][peak_idx_ecg]
    hr_ecg = peak_f_ecg * 60

    # ==========================================
    # 打印法庭对峙结果
    # ==========================================
    print("-" * 50)
    print(f"🎯 算法性能对决 (Hampel vs Raw vs Ground Truth):")
    print(f"   ❤️  真实心率 (ECG):    {hr_ecg:.1f} BPM")
    print(f"   💩  原始预测 (RAW):    {hr_raw:.1f} BPM  | 误差: {abs(hr_ecg - hr_raw):.1f} BPM")
    print(f"   ✨  滤波预测 (HAMPEL): {hr_hmp:.1f} BPM  | 误差: {abs(hr_ecg - hr_hmp):.1f} BPM")
    print("-" * 50)

    # ==========================================
    # 画图展示 (5 行硬核图表)
    # ==========================================
    plt.figure(figsize=(16, 8))

    # --- 1. 时域对比 (叠在一起看斩首效果) ---
    plt.subplot(1, 1, 1)
    plt.plot(signal_raw, color='gray', alpha=0.5, linewidth=2, label='Raw Signal (with spikes)')
    plt.plot(signal_hampel, color='blue', linewidth=1, label='Hampel Filtered')
    plt.title(f'[WIFI Time Domain] Hampel vs Raw: {target_col}')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # # --- 2. 频域：原始信号 ---
    # plt.subplot(5, 1, 2)
    # plt.plot(freqs_raw, mag_raw, color='gray', linewidth=1.5)
    # plt.plot(peak_f_raw, peak_m_raw, 'ro', markersize=8, label=f'Raw HR: {hr_raw:.1f} BPM')
    # plt.title('[WIFI Freq] RAW Spectrum (Notice the noise floor)')
    # plt.ylabel('Magnitude')
    # plt.xlim(0, 5)
    # plt.axvspan(0.8, 2.0, color='red', alpha=0.05, label='Heartbeat Range')
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)
    #
    # # --- 3. 频域：Hampel 滤波后 ---
    # plt.subplot(5, 1, 3)
    # plt.plot(freqs_hmp, mag_hmp, color='purple', linewidth=1.5)
    # plt.plot(peak_f_hmp, peak_m_hmp, 'ro', markersize=8, label=f'Hampel HR: {hr_hmp:.1f} BPM')
    # plt.title('[WIFI Freq] HAMPEL Filtered Spectrum (Cleaner peaks)')
    # plt.ylabel('Magnitude')
    # plt.xlim(0, 5)
    # plt.axvspan(0.8, 2.0, color='red', alpha=0.1)
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)
    #
    # # --- 4. 时域：ECG ---
    # plt.subplot(5, 1, 4)
    # plt.plot(label, color='red', linewidth=1)
    # plt.title('[ECG Time] Ground Truth Label')
    # plt.ylabel('Probability')
    # plt.grid(True, alpha=0.3)
    #
    # # --- 5. 频域：ECG ---
    # plt.subplot(5, 1, 5)
    # plt.plot(freqs_ecg, mag_ecg, color='darkorange', linewidth=1.5)
    # plt.plot(peak_f_ecg, peak_m_ecg, 'go', markersize=8, label=f'True HR: {hr_ecg:.1f} BPM')
    # plt.title('[ECG Freq] Ground Truth Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.xlim(0, 5)
    # plt.axvspan(0.8, 2.0, color='red', alpha=0.1)
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print(f"❌ 找不到列 {target_col}，请检查你的 CSV 内容！")