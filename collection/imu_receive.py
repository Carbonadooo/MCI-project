import serial
import csv
import time
import matplotlib.pyplot as plt
from collections import deque

SERIAL_PORT = "/dev/ttyACM0"       # Windows: COM口，Linux/Mac: /dev/ttyUSB0
BAUD_RATE = 115200
OUTPUT_CSV = "imu_data.csv"
REALTIME_PLOT = True       # 是否显示实时图

PLOT_WINDOW = 100          # 绘图显示最近N个点
PLOT_INTERVAL = 10          # 每接收多少行数据更新一次绘图
# -----------------------------

# 初始化串口
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # 等待串口稳定

# 打开 CSV 文件
csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp","ax","ay","az","gx","gy","gz","mx","my","mz"])

# 初始化绘图
if REALTIME_PLOT:
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    ax_data = [deque(maxlen=PLOT_WINDOW) for _ in range(9)]
    lines = []
    labels = ["ax","ay","az","gx","gy","gz","mx","my","mz"]
    colors = ["r","g","b","r","g","b","r","g","b"]
    for i in range(3):
        for j in range(3):
            line, = axes[i].plot([], [], colors[i*3+j], label=labels[i*3+j])
            lines.append(line)
        axes[i].legend()
        axes[i].grid(True)

start_time = time.time()
update_count = 0

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 9:
            continue  # 数据不完整，跳过

        # 转换成浮点数
        data = list(map(float, parts))
        timestamp = time.time() - start_time

        # 写入 CSV
        csv_writer.writerow([timestamp] + data)
        csv_file.flush()

        # 保存到队列
        for i in range(9):
            ax_data[i].append(data[i])

        # 控制绘图更新频率
        if REALTIME_PLOT:
            update_count += 1
            if update_count >= PLOT_INTERVAL:
                update_count = 0
                for i in range(9):
                    lines[i].set_xdata(range(len(ax_data[i])))
                    lines[i].set_ydata(ax_data[i])
                for ax_ in axes:
                    ax_.relim()
                    ax_.autoscale_view()
                plt.pause(0.001)

except KeyboardInterrupt:
    print("结束采集")
finally:
    csv_file.close()
    ser.close()
    if REALTIME_PLOT:
        plt.ioff()
        plt.show()
