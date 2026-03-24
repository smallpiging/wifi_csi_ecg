import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
import os
import pandas as pd
from ecgdetectors import Detectors
import pywt

SHOW_PLOT = True

class Process_dataset():
    def __init__(self, source_path='./saved_datasets', target_path='./processed_datasets', idx=0):
        self.fs = 125
        self.detectors = Detectors(self.fs)

        data_path = os.listdir(source_path)
        self.csv_path = [f for f in data_path if f.endswith('.csv') and f.startswith('yck')]
        # self.csv_path = self.csv_path[idx]
        self.pre_process_path = self.csv_path[idx]
        self.df = pd.read_csv(os.path.join(source_path, self.pre_process_path))
        self.target_path = target_path

    def del_df(self):
        # 删除phase信息
        phase_cols = [col for col in self.df.columns if 'phase' in col.lower()]
        if len(phase_cols) > 0:
            self.df.drop(columns=phase_cols, inplace=True)
            print(f"🔪 成功砍掉 {len(phase_cols)} 列相位数据，只保留幅值！")
        else:
            print("⚠️ 没找到带有 phase 字眼的列，请检查你的原始 CSV 列名哦！")

        # 删除其他信息
        self.df.drop(columns=['WIFI_Timestamp'], inplace=True)
        self.df.drop(columns=['WIFI_Mean_Mag'], inplace=True)
        self.df.drop(columns=['Aligned_Breath'], inplace=True)

    def process_csi_df(self):

        # 小波变换系数
        wavelet = 'sym5'
        level = 7  # 我们把信号过 7 层筛子

        mag_cols = [col for col in self.df.columns if 'mag' in col.lower()]
        for col in mag_cols:
            signal = self.df[col].values
            # 1. 准备信号：去除均值（消除直流偏置，防止波形飘在天上）
            s = signal - np.mean(signal)

            # 3. 核心分解：把信号倒进筛子！
            # 返回的 coeffs 是一个列表，结构为: [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
            coeffs = pywt.wavedec(s, wavelet, level=level)

            # ---------------------------------------------------------
            # 4. 提取心跳信号 (高频，剥离低频呼吸和超高频底噪)
            # ---------------------------------------------------------
            # 我们复制一份全空的筛子 (把所有层清零)
            coeffs_heart = [np.zeros_like(c) for c in coeffs]

            # 假设你的采样率是 100Hz，经过 7 层分解：
            # cD1, cD2, cD3 通常是高频环境噪声 (> 6Hz)
            # cD4, cD5, cD6 刚好对应大约 0.8Hz 到 6Hz 左右的频段，心跳就在这里！
            coeffs_heart[1] = coeffs[1]  # 保留 cD7
            coeffs_heart[2] = coeffs[2]  # 保留 cD6
            # coeffs_heart[3] = coeffs[3]  # 保留 cD5
            # coeffs_heart[4] = coeffs[4]  # 保留 cD4
            # coeffs_heart[5] = coeffs[5]  # 保留 cD3
            # coeffs_heart[6] = coeffs[6]  # 保留 cD2
            # 逆小波变换：拿着保留的心跳系数，倒推回时间波形
            heartbeat_clean = pywt.waverec(coeffs_heart, wavelet)
            self.df[col] = heartbeat_clean

            # ---------------------------------------------------------
            # 5. 提取呼吸信号 (低频，剥离高频心跳)
            # ---------------------------------------------------------
            coeffs_breath = [np.zeros_like(c) for c in coeffs]

            # 呼吸频率通常在 0.2Hz - 0.5Hz，属于非常低频的信号
            # 它们通常藏在 cA7 (最低频的近似分量) 和 cD7 (最低频的细节分量) 里
            coeffs_breath[0] = coeffs[0]  # 保留 cA7
            coeffs_breath[1] = coeffs[1]  # 保留 cD7

            # 逆小波变换：重构呼吸波形
            breath_clean = pywt.waverec(coeffs_breath, wavelet)

            # if SHOW_PLOT:
            #     plt.figure(figsize=(14, 8))
            #
            #     # 原始信号
            #     plt.subplot(4, 1, 1)
            #     plt.plot(s, label='Original CSI (Subcarrier 40)', color='gray', alpha=0.7)
            #     plt.title("Step 1: Original Raw Signal")
            #     plt.legend()
            #
            #     # 提取出的呼吸波
            #     plt.subplot(4, 1, 2)
            #     plt.plot(breath_clean, label='Extracted Respiration (Low Freq)', color='blue', linewidth=2)
            #     plt.title("Step 2: Clean Respiration Waveform")
            #     plt.legend()
            #
            #     # 提取出的心跳波
            #     plt.subplot(4, 1, 3)
            #     plt.plot(heartbeat_clean, label='Extracted Heartbeat (Mid-High Freq)', color='red', linewidth=1.5)
            #     plt.title("Step 3: Clean Heartbeat Waveform")
            #     plt.legend()
            #
            #     plt.subplot(4, 1, 4)
            #     ecg = self.df['ECG_Heatmap_Label']
            #     plt.plot(ecg, label='Aligned ECG_Heatmap_Label')
            #     plt.title("Step 4: Aligned ECG Waveform")
            #     plt.legend()
            #
            #     plt.tight_layout()
            #     plt.show()

    def process_df_label(self):
        self.df = self.df.iloc[250:].reset_index(drop=True)
        signal = self.df['Aligned_ECG'].values

        # 专业库处理
        # r_peaks = detectors.wqrs_detector(signal)
        # r_peaks1 = detectors.two_average_detector(signal)
        # print(r_peaks)
        # print(r_peaks1)

        # 自己写
        # fs=100Hz 时，0.4 秒 = 40 个点。所以 distance 设为 40，防止把 T 波误认为 R 峰。
        # prominence: 突起程度（根据你真实的 ECG 幅值调整，这里随便设个阈值，比如 50）
        r_peaks, _ = find_peaks(signal, distance=40, prominence=1.8)

        # 构建热力图
        probability_mask = self.generate_gaussian_mask(len(signal), r_peaks, sigma=4)
        self.df['Aligned_ECG'] = probability_mask
        self.df.rename(columns={'Aligned_ECG': 'ECG_Heatmap_Label'}, inplace=True)

        # 画图
        if SHOW_PLOT:
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 1, 1)
            plt.plot(signal)
            plt.scatter(r_peaks, signal[r_peaks], color='red')
            plt.subplot(2, 1, 2)
            plt.plot(probability_mask)
            plt.show()

    @staticmethod
    def generate_gaussian_mask(signal_length, peaks, sigma=3):
        """根据精准的 R 峰索引，生成 0-1 的高斯概率热力图"""
        mask = np.zeros(signal_length, dtype=float)
        x = np.arange(signal_length)
        for peak in peaks:
            bump = np.exp(-0.3 * ((x - peak) / sigma) ** 2)
            mask = np.maximum(mask, bump)
        return mask

    def save_df(self):
        save_path = os.path.join(self.target_path, self.pre_process_path)
        self.df.to_csv(save_path, index=False)

if __name__ == "__main__":
    process_dataset = Process_dataset(source_path='./saved_datasets', target_path='./processed_datasets', idx=1)
    process_dataset.process_df_label()
    process_dataset.del_df()
    process_dataset.process_csi_df()
    process_dataset.save_df()
