import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
import os
import pandas as pd
from ecgdetectors import Detectors
import pywt
from sklearn.decomposition import PCA  # 新增 PCA 导入

SHOW_PLOT = True


class Process_dataset():
    # 注意：我把 target_path 默认改成了 pca_datasets
    def __init__(self, source_path='./saved_datasets', target_path='./pca_datasets', idx=0):
        self.fs = 125
        self.detectors = Detectors(self.fs)

        data_path = os.listdir(source_path)
        # 过滤出符合条件的 csv 文件列表
        self.csv_files = [f for f in data_path if f.endswith('.csv') and f.startswith('yck')]

        if not self.csv_files:
            raise FileNotFoundError(f"在 {source_path} 目录下没有找到以 'yck' 开头的 csv 文件！")

        self.pre_process_path = self.csv_files[idx]
        self.df = pd.read_csv(os.path.join(source_path, self.pre_process_path))
        self.target_path = target_path

        # 如果目标文件夹不存在，自动创建它
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)
            print(f"📁 已自动创建 PCA 结果保存文件夹: {self.target_path}")

    def del_df(self):
        # 删除phase信息
        phase_cols = [col for col in self.df.columns if 'phase' in col.lower()]
        if len(phase_cols) > 0:
            self.df.drop(columns=phase_cols, inplace=True)
            print(f"🔪 成功砍掉 {len(phase_cols)} 列相位数据，只保留幅值！")
        else:
            print("⚠️ 没找到带有 phase 字眼的列，请检查原始 CSV 列名！")

        # 删除其他无关信息
        cols_to_drop = ['WIFI_Timestamp', 'WIFI_Mean_Mag', 'Aligned_Breath']
        for col in cols_to_drop:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)

    def process_csi_df(self):
        mag_cols = [col for col in self.df.columns if 'mag' in col.lower()]
        num_samples = len(self.df)

        # 创建一个空矩阵，用来装 64 个经过小波清洗的子载波
        clean_csi_matrix = np.zeros((num_samples, len(mag_cols)))

        # 小波变换参数
        wavelet = 'sym5'
        level = 7

        print(f"🌊 开始进行批量小波清洗，共 {len(mag_cols)} 个子载波...")
        for i, col in enumerate(mag_cols):
            signal = self.df[col].values
            # 1. 去除直流偏置
            s = signal - np.mean(signal)

            # 2. 小波分解
            coeffs = pywt.wavedec(s, wavelet, level=level)

            # 3. 提取心跳频段
            coeffs_heart = [np.zeros_like(c) for c in coeffs]
            coeffs_heart[1] = coeffs[1]  # cD7
            coeffs_heart[2] = coeffs[2]  # cD6
            # coeffs_heart[3] = coeffs[3]  # cD5

            # 4. 重构并强制截断长度 (极其关键，防止重构后多出1个点报错)
            heartbeat_clean = pywt.waverec(coeffs_heart, wavelet)[:num_samples]

            # 把洗干净的数据装进矩阵对应的列
            clean_csi_matrix[:, i] = heartbeat_clean

        # ==========================================
        # 核心加入 PCA 降维打击
        # ==========================================
        print("🎯 开始执行 PCA 主成分分析...")
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(clean_csi_matrix)

        pc1 = pcs[:, 0]
        pc2 = pcs[:, 1]
        pc3 = pcs[:, 2]

        print(f"📊 PCA 能量占比: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

        # 将原来的几十根 mag 原始数据列彻底从 DataFrame 里删掉
        self.df.drop(columns=mag_cols, inplace=True)

        # 把最纯净的 3 个主成分添加进 DataFrame
        self.df['PC1'] = pc1
        self.df['PC2'] = pc2
        self.df['PC3'] = pc3

        # 画图验证环节
        if SHOW_PLOT:
            plt.figure(figsize=(16, 12))

            # 画出第一主成分
            plt.subplot(4, 1, 1)
            plt.plot(pc1, label='PC1 (Primary Principal Component)', color='blue', linewidth=1.5)
            plt.title("PCA Result: PC1 (Contains strongest highly-correlated dynamics)")
            plt.legend()

            # 画出第二主成分
            plt.subplot(4, 1, 2)
            plt.plot(pc2, label='PC2 (Secondary Principal Component)', color='green', linewidth=1.5)
            plt.title("PCA Result: PC2")
            plt.legend()

            # 画出第三主成分
            plt.subplot(4, 1, 3)
            plt.plot(pc3, label='PC3 (Tertiary Principal Component)', color='orange', linewidth=1.5)
            plt.title("PCA Result: PC3")
            plt.legend()

            # 画出你精准对齐的心电 Ground Truth
            plt.subplot(4, 1, 4)
            ecg_label = self.df['ECG_Heatmap_Label']
            plt.plot(ecg_label, label='ECG Heatmap Label', color='red')
            plt.title("Ground Truth: ECG Aligned Spikes")
            plt.legend()

            plt.tight_layout()
            plt.show()

    def process_df_label(self):
        # 掐头去尾，舍弃刚开始不稳定的一段数据
        self.df = self.df.iloc[250:].reset_index(drop=True)
        signal = self.df['Aligned_ECG'].values

        # 提取 R 峰
        r_peaks, _ = find_peaks(signal, distance=40, prominence=1.8)

        # 构建热力图标签
        probability_mask = self.generate_gaussian_mask(len(signal), r_peaks, sigma=4)
        self.df['Aligned_ECG'] = probability_mask
        self.df.rename(columns={'Aligned_ECG': 'ECG_Heatmap_Label'}, inplace=True)

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
        # 保存到你指定的新文件夹 (pca_datasets)
        save_path = os.path.join(self.target_path, self.pre_process_path)
        self.df.to_csv(save_path, index=False)
        print(f"💾 数据已成功提纯并保存至: {save_path}")


if __name__ == "__main__":
    # idx 设为你想要处理的文件索引
    process_dataset = Process_dataset(source_path='./saved_datasets', target_path='./pca_datasets', idx=9)
    process_dataset.process_df_label()
    process_dataset.del_df()
    process_dataset.process_csi_df()
    process_dataset.save_df()