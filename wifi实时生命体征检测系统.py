import os
import queue
import time
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit
from collections import deque

from wifi_csi_ui import Ui_MainWindow
import serial
from threading import Thread, Lock
import pyqtgraph as pg
import numpy as np
from datetime import datetime
import pandas as pd
import zmq
import pywt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks


def hampel_filter_pandas(input_array, window_size=20, n_sigmas=3):
    series = pd.Series(input_array)
    rolling_median = series.rolling(window=2 * window_size + 1, center=True).median()
    MAD = 1.4826 * (series - rolling_median).abs().rolling(window=2 * window_size + 1, center=True).median()
    outlier_idx = (series - rolling_median).abs() > (n_sigmas * MAD)
    series_filtered = series.copy()
    series_filtered[outlier_idx] = rolling_median[outlier_idx]
    return series_filtered.bfill().ffill().values


def wavelet_denoise_heart(signal, wavelet='sym5', level=7):  # ⚠️ 改为了 level=5 以适配 5 秒窗长
    s = signal - np.mean(signal)
    coeffs = pywt.wavedec(s, wavelet, level=level)
    coeffs_heart = [np.zeros_like(c) for c in coeffs]
    # 假设 level=5, cD5~cD3 大致覆盖心跳频段
    coeffs_heart[1] = coeffs[1]
    coeffs_heart[2] = coeffs[2]
    coeffs_heart[3] = coeffs[3]

    return pywt.waverec(coeffs_heart, wavelet)[:len(signal)]


def wavelet_denoise_breath(signal, wavelet='sym5', level=7):
    """提取低频呼吸：只保留小波变换的最底层系数 (cA7, cD7)"""
    s = signal - np.mean(signal)
    coeffs = pywt.wavedec(s, wavelet, level=level)
    coeffs_breath = [np.zeros_like(c) for c in coeffs]

    # 呼吸是极低频信号，保留 0 和 1
    coeffs_breath[0] = coeffs[0]
    coeffs_breath[1] = coeffs[1]

    return pywt.waverec(coeffs_breath, wavelet)[:len(signal)]

def extract_pca_components(matrix, n_components=3):
    """
    PCA 降维提取器：把 52 根载波的矩阵，浓缩成几个核心主成分
    :param matrix: 清洗后的二维 CSI 矩阵 (时间点 x 子载波)
    :return: pc1, pc2, pc3 (三个最核心的一维信号)
    """
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(matrix)
    # 按照能量贡献度，返回前三个主成分
    return pcs[:, 0], pcs[:, 1], pcs[:, 2]


def calculate_heart_rate(signal, fs=100, low_f=0.8, high_f=2.0):
    """
    FFT 心率提取器：对一维信号进行频域变换，并在指定红区内寻找最大峰值
    :param signal: 一维数组 (通常是 PC1)
    :param fs: 采样率
    :param low_f: 心跳频段下限 (默认 0.8 Hz)
    :param high_f: 心跳频段上限 (默认 2.0 Hz)
    :return: 预测的心率数值 (BPM)
    """
    N = len(signal)

    # 1. 去除直流分量 (基线漂移)
    signal_centered = signal - np.mean(signal)

    # 2. 计算 FFT 并生成频率轴
    fft_y = np.fft.fft(signal_centered)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    # 3. 截取正半轴
    half_n = N // 2
    pos_freqs = freqs[:half_n]
    pos_fft_mag = np.abs(fft_y[:half_n]) * 2 / N

    # 4. 锁定目标频段 (心跳红区)
    band_indices = np.where((pos_freqs >= low_f) & (pos_freqs <= high_f))[0]

    # 防御机制：如果给的频段太离谱，连一个点都没有
    if len(band_indices) == 0:
        return 0.0, pos_freqs, pos_fft_mag

    target_freqs = pos_freqs[band_indices]
    target_mags = pos_fft_mag[band_indices]

    # 5. 寻峰逻辑 (带突出度检测)
    many_peak_idx, _ = find_peaks(target_mags, distance=2, prominence=0.015)

    if many_peak_idx.size > 0:
        peak_values = target_mags[many_peak_idx]
        max_relative_idx = np.argmax(peak_values)
        peak_idx = many_peak_idx[max_relative_idx]
    else:
        # 降级方案：如果没有显著峰，直接抓取最高点
        peak_idx = np.argmax(target_mags)

    # 6. 计算最终心率 (BPM)
    pred_freq = target_freqs[peak_idx]
    pred_hr = pred_freq * 60

    return pred_hr, pos_freqs, pos_fft_mag

def calculate_breath_rate(signal, fs=100, low_f=0.1, high_f=0.5):
    """
    FFT 心率提取器：对一维信号进行频域变换，并在指定红区内寻找最大峰值
    :param signal: 一维数组 (通常是 PC1)
    :param fs: 采样率
    :param low_f: 心跳频段下限 (默认 0.8 Hz)
    :param high_f: 心跳频段上限 (默认 2.0 Hz)
    :return: 预测的心率数值 (BPM)
    """
    N = len(signal)

    # 🌟 补零魔法：强行扩展到 60 秒的数据长度
    virtual_length = fs * 120  # 125 * 60 = 7500 个点

    # 1. 去除直流分量 (基线漂移)
    signal_centered = signal - np.mean(signal)

    # 2. 计算 FFT 并生成频率轴
    fft_y = np.fft.fft(signal_centered, n=virtual_length)
    freqs = np.fft.fftfreq(virtual_length, d=1 / fs)

    # 3. 截取正半轴
    half_n = virtual_length // 2
    pos_freqs = freqs[:half_n]
    pos_fft_mag = np.abs(fft_y[:half_n]) * 2 / N

    # 4. 锁定目标频段 (心跳红区)
    band_indices = np.where((pos_freqs >= low_f) & (pos_freqs <= high_f))[0]

    # 防御机制：如果给的频段太离谱，连一个点都没有
    if len(band_indices) == 0:
        return 0.0, pos_freqs, pos_fft_mag

    target_freqs = pos_freqs[band_indices]
    target_mags = pos_fft_mag[band_indices]

    # 5. 寻峰逻辑 (带突出度检测)
    many_peak_idx, _ = find_peaks(target_mags, distance=2, prominence=0.015)

    if many_peak_idx.size > 0:
        peak_values = target_mags[many_peak_idx]
        max_relative_idx = np.argmax(peak_values)
        peak_idx = many_peak_idx[max_relative_idx]
    else:
        # 降级方案：如果没有显著峰，直接抓取最高点
        peak_idx = np.argmax(target_mags)

    # 6. 计算最终心率 (BPM)
    pred_freq = target_freqs[peak_idx]
    pred_hr = pred_freq * 60

    return pred_hr, pos_freqs, pos_fft_mag

# def calculate_heart_rate(signal, fs=125, low_f=0.8, high_f=2.0):
#     N = len(signal)
#
#     # 🌟 补零魔法：强行扩展到 60 秒的数据长度
#     virtual_length = fs * 20  # 125 * 60 = 7500 个点
#
#     signal_centered = signal - np.mean(signal)
#
#     # 用补零后的长度计算 FFT
#     fft_y = np.fft.fft(signal_centered, n=virtual_length)
#     freqs = np.fft.fftfreq(virtual_length, d=1 / fs)
#
#     half_n = virtual_length // 2
#     pos_freqs = freqs[:half_n]
#     # 注意：幅度依然除以原始真实长度 N，这样幅度数值才是准的
#     pos_fft_mag = np.abs(fft_y[:half_n]) * 2 / N
#
#     band_indices = np.where((pos_freqs >= low_f) & (pos_freqs <= high_f))[0]
#
#     if len(band_indices) == 0:
#         return 0.0, pos_freqs, pos_fft_mag
#
#     target_freqs = pos_freqs[band_indices]
#     target_mags = pos_fft_mag[band_indices]
#
#     many_peak_idx, _ = find_peaks(target_mags, distance=2, prominence=0.015)
#
#     if many_peak_idx.size > 0:
#         peak_values = target_mags[many_peak_idx]
#         max_relative_idx = np.argmax(peak_values)
#         peak_idx = many_peak_idx[max_relative_idx]
#     else:
#         peak_idx = np.argmax(target_mags)
#
#     pred_freq = target_freqs[peak_idx]
#     pred_hr = pred_freq * 60
#
#     return pred_hr, pos_freqs, pos_fft_mag

class MySignals(QObject):
    plain_text_print = pyqtSignal(QTextEdit, str)
    update_table = pyqtSignal(str)

class Stats(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加锁，用于画图
        self.data_lock = Lock()

        # 加载ui界面与初始化
        self.subcarries_idx = 0
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ms = MySignals()

        # 绑定按键等事件
        self.ms.plain_text_print.connect(self.printToGui)
        self.ui.sub_carries_idx.currentIndexChanged.connect(self.handle_update_table)

        # 初始化 ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.tcp_address = "tcp://127.0.0.1:55555"
        self.socket.connect(self.tcp_address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        print(f"✅ ZMQ 连接成功！{self.tcp_address}")

        # 核心参数
        self.fs = 100
        self.num_subcarriers = 52
        self.sec_per_process = 30  # 窗口长度: 15秒
        self.sec_per_step = 1      # 滑动步长: 2秒

        self.points_per_process = self.fs * self.sec_per_process
        self.points_per_step = self.fs * self.sec_per_step

        self.chunk_size = self.points_per_process * self.num_subcarriers
        self.step_size = self.points_per_step * self.num_subcarriers

        self.buffer = np.array([], dtype=np.complex64)

        # 展示数组
        self.x_data = np.linspace(0, self.sec_per_process, self.points_per_process)
        self.show_csi_data = np.zeros(self.points_per_process)      # 第一行的csi展示
        self.show_hampel_filter = np.zeros(self.points_per_process) # 第二行的滤波展示
        self.show_DWT_data = np.zeros(self.points_per_process)      # 第三行的DWT展示
        self.show_PAC_data = np.zeros(self.points_per_process)      # 第四行的PAC展示

        # FFT 数据数组 (因为去掉了负频率，长度是原来的一半 625 // 2 = 312)
        fft_len = (self.sec_per_process * self.fs) // 2
        self.show_FFT_freqs = np.zeros(fft_len)
        self.show_FFT_mags = np.zeros(fft_len)

        # 画出现在和历史图画
        pen = pg.mkPen(color='red', width=2)
        self.curve_csi_data = self.ui.wifi_csi.plot(self.x_data, self.show_csi_data, pen=pen)
        pen = pg.mkPen(color='green', width=2)
        self.curve_hampel_csi_data = self.ui.hampel_filter_csi.plot(self.x_data, self.show_hampel_filter, pen=pen)
        pen = pg.mkPen(color='red', width=2)
        self.curve_DWT_data = self.ui.DWT_csi.plot(self.x_data, self.show_DWT_data, pen=pen)
        pen = pg.mkPen(color='green', width=2)
        self.curve_PCA_data = self.ui.PCA_csi.plot(self.x_data, self.show_PAC_data, pen=pen)

        # 画 FFT 的画笔
        pen_fft = pg.mkPen(color='cyan', width=1.5)
        self.curve_FFT_data = self.ui.FFT.plot(self.show_FFT_freqs, self.show_FFT_mags, pen=pen_fft)
        # 可选优化：只展示 0~5Hz 的关键频段，让心跳峰值更明显
        self.ui.FFT.setXRange(0, 5)


        # 启动高频定时器 (替代原来的 while True), 每 10 毫秒醒来一次，去 ZMQ 管道里看有没有新数据
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(200)

    def printToGui(self, fb, text):
        fb.append(str(text))
        fb.ensureCursorVisible()

    def handle_update_table(self):
        self.subcarries_idx = int(self.ui.sub_carries_idx.currentText())

    def update_plot(self):
        with self.data_lock:
            show_csi_data = self.show_csi_data
            show_hampel_filter = self.show_hampel_filter
            show_DWT_data = self.show_DWT_data
            show_PAC_data = self.show_PAC_data

            # 取出 FFT 数组
            show_FFT_freqs = self.show_FFT_freqs
            show_FFT_mags = self.show_FFT_mags
        self.curve_csi_data.setData(self.x_data, show_csi_data)
        self.curve_hampel_csi_data.setData(self.x_data, show_hampel_filter)
        self.curve_DWT_data.setData(self.x_data, show_DWT_data)
        self.curve_PCA_data.setData(self.x_data, show_PAC_data)
        self.curve_FFT_data.setData(show_FFT_freqs, show_FFT_mags)

    def update_data(self):
        """核心调度器：疯狂吸水 + 满足条件就开炉炼丹"""
        # --- 阶段一：非阻塞式吸水 ---
        try:
            # 疯狂榨干管道里的所有暂存包，直到抛出 Again 异常
            while True:
                # 极其关键：flags=zmq.NOBLOCK 保证没有数据时不会卡死 GUI！
                raw_bytes = self.socket.recv(flags=zmq.NOBLOCK)
                data_array = np.frombuffer(raw_bytes, dtype=np.complex64)
                csi_array = data_array.reshape((-1, self.num_subcarriers))
                csi_data = np.abs(csi_array)[:,self.subcarries_idx]
                with self.data_lock:
                    self.show_csi_data = np.concatenate((self.show_csi_data, csi_data))
                    self.show_csi_data = self.show_csi_data[-self.points_per_process:]
                self.buffer = np.concatenate((self.buffer, data_array))
        except zmq.Again:
            # 管道吸干了，跳出循环，继续执行下面的逻辑
            pass
        # print('hello')
        # --- 阶段二：开闸放水，执行算法 ---
        if len(self.buffer) >= self.chunk_size:
            process_data = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.step_size:]  # 截留剩下的数据

            start_time = time.time()

            csi_complex_matrix = process_data.reshape((self.points_per_process, self.num_subcarriers))  # 重塑csi矩阵
            mag_matrix = np.abs(csi_complex_matrix)         # 变成幅度
            clean_matrix_heart = np.zeros_like(mag_matrix)
            clean_matrix_breath = np.zeros_like(mag_matrix)

            # 遍历清洗 52 个子载波
            for i in range(self.num_subcarriers):
                raw_signal = mag_matrix[:, i]
                hampel_signal = hampel_filter_pandas(raw_signal, window_size=20)
                clean_heart_signal = wavelet_denoise_heart(hampel_signal, level=7)
                clean_breath_signal = wavelet_denoise_breath(hampel_signal, level=7)
                if i == self.subcarries_idx:
                    with self.data_lock:
                        self.show_hampel_filter = hampel_signal
                        self.show_DWT_data = clean_heart_signal
                clean_matrix_heart[:, i] = clean_heart_signal
                clean_matrix_breath[:, i] = clean_breath_signal

            pc1, pc2, pc3 = extract_pca_components(clean_matrix_heart)
            pred_hr, fft_freqs, fft_mags = calculate_heart_rate(pc1, fs=self.fs, low_f=1.0, high_f=2.0)

            pc1_b, pc2_b, pc3_b = extract_pca_components(clean_matrix_breath)
            pred_br, _, _ = calculate_breath_rate(pc1_b, fs=self.fs, low_f=0.1, high_f=0.5)
            with self.data_lock:
                self.show_PAC_data = pc1
                self.show_FFT_freqs = fft_freqs
                self.show_FFT_mags = fft_mags


            self.ms.plain_text_print.emit(self.ui.hr_text, f"{pred_hr:.1f} RPM")
            self.ms.plain_text_print.emit(self.ui.rr, f"{pred_br:.1f} RPM")

            cost_time = time.time() - start_time
            print(f"✅ 处理完毕！耗时: {cost_time:.3f} 秒。")
        #
        #     # --- 阶段三：刷新图表 ---
        #     # 屏幕上同时画 52 根线会卡成乱码，我们挑一根看着顺眼的子载波画出来 (比如第 10 根)
        #     display_signal = clean_matrix_heart[:, 10]
        #
        #     # 更新 PyQtGraph 的数据
        #     self.plot_line.setData(display_signal)



if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.show()
    app.exec_()