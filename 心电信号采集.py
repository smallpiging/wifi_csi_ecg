import os
import queue
import time
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit
from collections import deque
from ecg_ui import Ui_ecg_get
import serial
from threading import Thread, Lock
import pyqtgraph as pg
import numpy as np
from datetime import datetime
import pandas as pd
from gnuradio import gr

import shared_data
from wifi_rx import wifi_rx

class MySignals(QObject):
    plain_text_print = pyqtSignal(QTextEdit, str)
    update_table = pyqtSignal(str)

class Stats(QMainWindow):
    def __init__(self):
        super().__init__()
        # 加锁，用于画图
        self.data_lock = Lock()

        # 加载ui界面与初始化
        self.thread = None
        self.ms = MySignals()
        self.ser = None
        self.ui = Ui_ecg_get()
        self.ui.setupUi(self)
        self.tb = None
        self.csi_buffer = deque(maxlen=10000)

        #绑定按键事件
        self.ui.start_serial.clicked.connect(self.handle_start_serial)
        self.ui.stop_serial.clicked.connect(self.handle_stop_serial)
        self.ui.save_data.clicked.connect(self.handle_save_data)
        self.ui.del_data.clicked.connect(self.handle_del_data)
        self.ui.load_list.clicked.connect(self.handle_load_list)
        self.ui.file_list.clicked.connect(self.handle_load_data)
        self.ui.init_usrp.clicked.connect(self.handle_init_usrp)
        self.ms.plain_text_print.connect(self.printToGui)

        # 画图点数，数据及画笔初始化
        self.sample_rate = 250
        self.window_size = 20
        self.max_points = self.sample_rate * self.window_size





        self.wifi_rate = 125
        self.wifi_max_points = self.wifi_rate * self.window_size

        self.x_data = np.linspace(0, self.window_size, self.max_points)
        self.x_wifi_data = np.linspace(0, self.window_size, self.wifi_max_points)
        self.ecg_data, self.breath_data = np.zeros(self.max_points), np.zeros(self.max_points)
        self.wifi_data = np.zeros(self.wifi_max_points)
        self.timestamp_ecg_data = np.zeros(self.max_points)
        self.timestamp_wifi_data = np.zeros(self.wifi_max_points)

        # 画出现在和历史图画
        pen = pg.mkPen(color='red', width=2)
        self.curve_ecg = self.ui.real_time_ecg_plot.plot(self.x_data, self.ecg_data, pen=pen)
        self.hist_curve_ecg = self.ui.history_ecg_plot.plot(pen=pen)
        pen = pg.mkPen(color='blue', width=2)
        self.curve_breath = self.ui.real_time_breath_plot.plot(self.x_data, self.breath_data, pen=pen)
        self.hist_curve_breath = self.ui.history_breath_plot.plot(pen=pen)
        pen = pg.mkPen(color='green', width=2)
        self.curve_wifi = self.ui.real_time_wifi_plot.plot(self.x_wifi_data, self.wifi_data, pen=pen)
        self.hist_curve_wifi = self.ui.history_wifi_plot.plot(pen=pen)

        # 保存文件夹
        self.save_dir = "saved_datasets"
        os.makedirs(self.save_dir, exist_ok=True)  # 如果文件夹不存在则自动创建

        # 图标更新策略
        pg.setConfigOptions(antialias=True) # 开启抗锯齿，线条更平滑，但会消耗一点点性能
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def printToGui(self, fb, text):
        fb.append(str(text))
        fb.ensureCursorVisible()

    def update_plot(self):
        with self.data_lock:
            ecg_snap = self.ecg_data.copy()
            breath_snap = self.breath_data.copy()
            wifi_snap = self.wifi_data.copy()
        self.curve_ecg.setData(self.x_data, ecg_snap)
        self.curve_breath.setData(self.x_data, breath_snap)
        self.curve_wifi.setData(self.x_wifi_data, wifi_snap)

    def handle_init_usrp(self):
        try:
            self.ms.plain_text_print.emit(self.ui.conf_text, 'initing usrp, waiting')
            time.sleep(0.1)
            self.tb = wifi_rx()
            self.ms.plain_text_print.emit(self.ui.conf_text, 'success to init')
        except Exception as e:
            self.tb = None
            self.ms.plain_text_print.emit(self.ui.conf_text, 'failed to init')

    def handle_start_serial(self): #打开串口
        try:
            #第一步连接串口
            self.ser = serial.Serial(port=self.ui.serial_numcb.currentText(),
                                     baudrate=eval(self.ui.baud_cb.currentText()), timeout=None)
            #第二部失效其他几个选项
            self.ui.start_serial.setEnabled(False)
            self.ui.stop_serial.setEnabled(True)
            self.ui.serial_numcb.setEnabled(False)
            self.ui.baud_cb.setEnabled(False)
            self.ui.save_data.setEnabled(False)
            self.ui.del_data.setEnabled(False)
            self.ms.plain_text_print.emit(self.ui.conf_text, f'Successfully open {self.ser.name}, Baud rate: {self.ser.baudrate}')

            if self.tb is not None:
                self.tb.start()
            else:
                self.ms.plain_text_print.emit(self.ui.conf_text, "failed init usrp")
            self.timer.start(200)

            def threadFunc():
                while self.ser.is_open:
                    try:
                        # 解码数据
                        line = self.ser.read_until(b'\n')
                        part = line.decode('utf-8', 'ignore').strip()
                        # self.ms.plain_text_print.emit(self.ui.serial_plain_text, part) # 打印数据
                        ecg_new_data = part.split(',')[0]
                        breath_new_data = part.split(',')[1]
                        # 解码时间
                        current_time_ecg = datetime.now().timestamp()
                        # 解码后加入数组
                        with self.data_lock:
                            self.ecg_data[:-1] = self.ecg_data[1:]
                            self.ecg_data[-1] = float(ecg_new_data)
                            self.breath_data[:-1] = self.breath_data[1:]
                            self.breath_data[-1] = float(breath_new_data)
                            self.timestamp_ecg_data[:-1] = self.timestamp_ecg_data[1:]
                            self.timestamp_ecg_data[-1] = current_time_ecg
                    except: #避免报错用的
                        pass
            self.thread = Thread(target=threadFunc)
            self.thread.start()

            def csi_thread_func():
                while self.tb is not None:
                    try:
                        ts, csi_matrix = shared_data.csi_queue.get(timeout=0.01)
                        mag_batch = np.mean(np.abs(csi_matrix))
                        self.csi_buffer.append((ts, csi_matrix))
                        with self.data_lock:
                            self.wifi_data[:-1] = self.wifi_data[1:]
                            self.wifi_data[-1:] = mag_batch
                            self.timestamp_wifi_data[:-1] = self.timestamp_wifi_data[1:]
                            self.timestamp_wifi_data[-1:] = ts
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(e)

            self.thread = Thread(target=csi_thread_func)
            self.thread.start()
        except Exception as e:
            self.ms.plain_text_print.emit(self.ui.conf_text, f'open serial failed\n,{e}')
            print('open serial failed\n', e)

    def handle_stop_serial(self): #关闭串口
        try:
            # 第一步关闭usrp及计数器
            if self.tb is not None:
                self.tb.stop()
                self.tb.wait()
                self.tb = None
            # 第二步关闭串口及计数器
            self.ser.close()
            self.timer.stop()
            # 第三部失效其他几个选项
            self.ui.start_serial.setEnabled(True)
            self.ui.stop_serial.setEnabled(False)
            self.ui.serial_numcb.setEnabled(True)
            self.ui.baud_cb.setEnabled(True)
            self.ui.save_data.setEnabled(True)
            self.ui.del_data.setEnabled(True)
            self.ms.plain_text_print.emit(self.ui.conf_text, 'close serial success')
        except Exception as e:
            self.ms.plain_text_print.emit(self.ui.conf_text, f'stop serial failed\n,{e}')
            print('stop serial failed\n', e)

    def handle_save_data(self):
        # 获取数据
        ecg_snap = self.ecg_data.copy()
        breath_snap = self.breath_data.copy()
        time_ecg_snap = self.timestamp_ecg_data.copy()
        # time_ecg_snap = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-2] for ts in time_ecg_snap]

        wifi_snap = self.wifi_data.copy()
        time_wifi_snap = self.timestamp_wifi_data.copy()
        # time_wifi_snap = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-2] for ts in time_wifi_snap]
        csi_buffer_snap = list(self.csi_buffer)


        if len(csi_buffer_snap) > 0 and isinstance(csi_buffer_snap[0], tuple):
            buffer_times = np.array([item[0] for item in csi_buffer_snap])
            buffer_frames = np.array([item[1] for item in csi_buffer_snap])
        else:
            buffer_times = np.array([])
            buffer_frames = np.zeros((0, 52))

        # 准备用于存放对齐后数据的空列表
        aligned_ecg = []
        aligned_breath = []
        aligned_csi_mag_52 = []
        aligned_csi_phase_52 = []

        # 4. 核心对齐算法：以 WiFi 的时间戳为基准，遍历寻找最近的邻居
        for t_w in time_wifi_snap:
            # 寻找时间线上离 t_w 最近的 ECG/Breath 数据点
            if len(time_ecg_snap[time_ecg_snap > 0]) > 0:
                ecg_idx = np.abs(time_ecg_snap - t_w).argmin()
                aligned_ecg.append(ecg_snap[ecg_idx])
                aligned_breath.append(breath_snap[ecg_idx])
            else:
                aligned_ecg.append(0)
                aligned_breath.append(0)

            # 寻找时间线上离 t_w 最近的 52 载波底层数据
            if len(buffer_times) > 0:
                csi_idx = np.abs(buffer_times - t_w).argmin()
                # 这里我们取复数的绝对值 (幅值 Magnitude) 进行保存，方便后续模型读取
                # 如果你需要保存相位，可以将 np.abs 换成 np.angle
                aligned_csi_mag_52.append(np.abs(buffer_frames[csi_idx]))
                aligned_csi_phase_52.append(np.angle(buffer_frames[csi_idx]))

            else:
                aligned_csi_mag_52.append(np.zeros(52))
                aligned_csi_phase_52.append(np.zeros(52))

        # 转换为 NumPy 矩阵方便操作，此时它的 shape 是 (N, 52)
        aligned_csi_mag_52 = np.array(aligned_csi_mag_52)
        aligned_csi_phase_52 = np.array(aligned_csi_phase_52)

        time_wifi_snap = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in time_wifi_snap]

        # 利用 pandas 构建数据集，准备写入CSV
        data_dict = {
            'WIFI_Timestamp': time_wifi_snap,
            'Aligned_ECG': aligned_ecg,
            'Aligned_Breath': aligned_breath,
            'WIFI_Mean_Mag': wifi_snap  # 在界面上画图用的那根平均折线
        }

        for i in range(52):
            data_dict[f'CSI_Mag_{i}'] = aligned_csi_mag_52[:, i]
            data_dict[f'CSI_Phase_{i}'] = aligned_csi_phase_52[:, i]

        df = pd.DataFrame(data_dict)

        # 生成以当前时间命名的文件名
        file_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(self.save_dir, file_name)

        df.to_csv(file_path, index=False, encoding='utf-8-sig') # 保存到csv文件中
        self.ms.plain_text_print.emit(self.ui.conf_text, f'Save data successfully')

    def handle_del_data(self):
        self.ui.conf_text.clear()
        self.ui.serial_plain_text.clear()

        # 图标重新画
        self.ecg_data, self.breath_data = np.zeros(self.max_points), np.zeros(self.max_points)
        self.wifi_data = np.zeros(self.wifi_max_points)

        self.timestamp_ecg_data = np.zeros(self.max_points)
        self.timestamp_wifi_data = np.zeros(self.wifi_max_points)
        self.csi_buffer.clear()

        self.curve_ecg.setData(self.x_data, self.ecg_data)
        self.curve_breath.setData(self.x_data, self.breath_data)
        self.curve_wifi.setData(self.x_wifi_data, self.wifi_data)

    def handle_load_list(self):
        self.ui.file_list.clear()
        file_list = os.listdir(self.save_dir)
        csv_file = [f for f in file_list if os.path.splitext(f)[1] == '.csv']
        self.ui.file_list.addItems(csv_file)
        self.ms.plain_text_print.emit(self.ui.conf_text, f'Load list successfully')

    def handle_load_data(self):
        csv_file = self.ui.file_list.currentItem().text()
        df = pd.read_csv(os.path.join(self.save_dir, csv_file))
        # if 'ECG' in df.columns and 'Breath' in df.columns:
        #     hist_ecg = df['ECG'].values
        #     hist_breath = df['Breath'].values
        #
        #     x_hist = np.linspace(0, self.window_size, len(hist_ecg))
        #
        #     self.hist_curve_ecg.setData(x_hist, hist_ecg)
        #     self.hist_curve_breath.setData(x_hist, hist_breath)
        #
        #     self.ms.plain_text_print.emit(self.ui.conf_text, f'Displaying History: {csv_file}')

        if 'Aligned_ECG' in df.columns and 'Aligned_Breath' in df.columns and 'WIFI_Mean_Mag' in df.columns:
            hist_ecg = df['Aligned_ECG'].values
            hist_breath = df['Aligned_Breath'].values
            hist_wifi = df['WIFI_Mean_Mag'].values

            x_hist = np.linspace(0, self.window_size, len(hist_ecg))

            self.hist_curve_ecg.setData(x_hist, hist_ecg)
            self.hist_curve_breath.setData(x_hist, hist_breath)
            self.hist_curve_wifi.setData(x_hist, hist_wifi)

            self.ms.plain_text_print.emit(self.ui.conf_text, f'Displaying History: {csv_file}')

if __name__ == '__main__':
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("警告: 无法开启实时调度 (可能需要 root/sudo 权限)。")
    app = QApplication([])
    stats = Stats()
    stats.show()
    app.exec_()