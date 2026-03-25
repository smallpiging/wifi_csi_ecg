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

class MySignals(QObject):
    plain_text_print = pyqtSignal(QTextEdit, str)
    update_table = pyqtSignal(str)

class Stats(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载ui界面与初始化
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ms = MySignals()

        # 绑定按键等事件
        self.ms.plain_text_print.connect(self.printToGui)

        # 初始化 ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.tcp_address = "tcp://127.0.0.1:55555"
        self.socket.connect(self.tcp_address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        print(f"✅ ZMQ 连接成功！{self.tcp_address}")

        # 核心参数
        self.fs = 125
        self.num_subcarriers = 52
        self.sec_per_process = 5
        self.points_per_process = self.fs * self.sec_per_process
        self.chunk_size = self.points_per_process * self.num_subcarriers
        self.buffer = np.array([], dtype=np.complex64)

        # 启动高频定时器 (替代原来的 while True), 每 10 毫秒醒来一次，去 ZMQ 管道里看有没有新数据
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)

    def printToGui(self, fb, text):
        fb.append(str(text))
        fb.ensureCursorVisible()

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
        print('hello')
        # --- 阶段二：开闸放水，执行算法 ---
        # if len(self.buffer) >= self.chunk_size:
        #     process_data = self.buffer[:self.chunk_size]
        #     self.buffer = self.buffer[self.chunk_size:]  # 截留剩下的数据
        #
        #     start_time = time.time()
        #     # print("🚀 [触发] 攒够5秒数据，开始执行 52 载波并行滤波...")
        #
        #     csi_complex_matrix = process_data.reshape((self.points_per_process, self.num_subcarriers))
        #     mag_matrix = np.abs(csi_complex_matrix)
        #     clean_matrix = np.zeros_like(mag_matrix)
        #
        #     # 遍历清洗 52 个子载波
        #     for i in range(self.num_subcarriers):
        #         raw_signal = mag_matrix[:, i]
        #         hampel_signal = hampel_filter_pandas(raw_signal, window_size=20)
        #         clean_signal = wavelet_denoise(hampel_signal, level=5)
        #         clean_matrix[:, i] = clean_signal
        #
        #     cost_time = time.time() - start_time
        #     print(f"✅ 处理完毕！耗时: {cost_time:.3f} 秒。")
        #
        #     # --- 阶段三：刷新图表 ---
        #     # 屏幕上同时画 52 根线会卡成乱码，我们挑一根看着顺眼的子载波画出来 (比如第 10 根)
        #     display_signal = clean_matrix[:, 10]
        #
        #     # 更新 PyQtGraph 的数据
        #     self.plot_line.setData(display_signal)



if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.show()
    app.exec_()