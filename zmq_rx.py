import zmq
import numpy as np
import pandas as pd
import pywt
import time
import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore


# ==========================================
# 算法组件库保持不变
# ==========================================
def hampel_filter_pandas(input_array, window_size=20, n_sigmas=3):
    series = pd.Series(input_array)
    rolling_median = series.rolling(window=2 * window_size + 1, center=True).median()
    MAD = 1.4826 * (series - rolling_median).abs().rolling(window=2 * window_size + 1, center=True).median()
    outlier_idx = (series - rolling_median).abs() > (n_sigmas * MAD)
    series_filtered = series.copy()
    series_filtered[outlier_idx] = rolling_median[outlier_idx]
    return series_filtered.bfill().ffill().values


def wavelet_denoise(signal, wavelet='sym5', level=5):  # ⚠️ 改为了 level=5 以适配 5 秒窗长
    s = signal - np.mean(signal)
    coeffs = pywt.wavedec(s, wavelet, level=level)
    coeffs_heart = [np.zeros_like(c) for c in coeffs]

    # 假设 level=5, cD5~cD3 大致覆盖心跳频段
    if len(coeffs) > 3:
        coeffs_heart[1] = coeffs[1]
        coeffs_heart[2] = coeffs[2]
        coeffs_heart[3] = coeffs[3]

    return pywt.waverec(coeffs_heart, wavelet)[:len(signal)]


# ==========================================
# 实时绘图与接收主窗口
# ==========================================
class RealTimePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WiFi CSI 实时生命体征监测雷达")
        self.resize(1000, 600)

        # 1. 初始化 PyQtGraph 图表
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground('k')  # 极客黑背景
        self.graphWidget.setTitle("Real-time Processed CSI (Subcarrier 10)", color="w", size="15pt")
        self.graphWidget.showGrid(x=True, y=True, alpha=0.3)

        # 创建一条画图的线 (青色)
        self.plot_line = self.graphWidget.plot([], [], pen=pg.mkPen(color=(0, 255, 255), width=2))

        # 2. 初始化 ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.tcp_address = "tcp://127.0.0.1:55555"
        self.socket.connect(self.tcp_address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        print(f"✅ ZMQ 连接成功！{self.tcp_address}")

        # 3. 核心参数
        self.fs = 125
        self.num_subcarriers = 52
        self.sec_per_process = 5
        self.points_per_process = self.fs * self.sec_per_process
        self.chunk_size = self.points_per_process * self.num_subcarriers

        self.buffer = np.array([], dtype=np.complex64)

        # 4. 启动高频定时器 (替代原来的 while True)
        # 每 10 毫秒醒来一次，去 ZMQ 管道里看有没有新数据
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)

    def update_data(self):
        """核心调度器：疯狂吸水 + 满足条件就开炉炼丹"""
        # --- 阶段一：非阻塞式吸水 ---
        try:
            # 疯狂榨干管道里的所有暂存包，直到抛出 Again 异常
            while True:
                # 极其关键：flags=zmq.NOBLOCK 保证没有数据时不会卡死 GUI！
                raw_bytes = self.socket.recv(flags=zmq.NOBLOCK)
                data_array = np.frombuffer(raw_bytes, dtype=np.complex64)
                self.buffer = np.concatenate((self.buffer, data_array))
        except zmq.Again:
            # 管道吸干了，跳出循环，继续执行下面的逻辑
            pass

        # --- 阶段二：开闸放水，执行算法 ---
        if len(self.buffer) >= self.chunk_size:
            process_data = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]  # 截留剩下的数据

            start_time = time.time()
            # print("🚀 [触发] 攒够5秒数据，开始执行 52 载波并行滤波...")

            csi_complex_matrix = process_data.reshape((self.points_per_process, self.num_subcarriers))
            mag_matrix = np.abs(csi_complex_matrix)
            clean_matrix = np.zeros_like(mag_matrix)

            # 遍历清洗 52 个子载波
            for i in range(self.num_subcarriers):
                raw_signal = mag_matrix[:, i]
                hampel_signal = hampel_filter_pandas(raw_signal, window_size=20)
                clean_signal = wavelet_denoise(hampel_signal, level=5)
                clean_matrix[:, i] = clean_signal

            cost_time = time.time() - start_time
            print(f"✅ 处理完毕！耗时: {cost_time:.3f} 秒。")

            # --- 阶段三：刷新图表 ---
            # 屏幕上同时画 52 根线会卡成乱码，我们挑一根看着顺眼的子载波画出来 (比如第 10 根)
            display_signal = clean_matrix[:, 10]

            # 更新 PyQtGraph 的数据
            self.plot_line.setData(display_signal)

    def closeEvent(self, event):
        """关闭窗口时的安全清理工作"""
        print("\n🛑 窗口关闭，正在断开 ZMQ 管道...")
        self.timer.stop()
        self.socket.close()
        self.context.term()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = RealTimePlotter()
    main_window.show()
    sys.exit(app.exec_())