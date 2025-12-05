import sys
import os
import time
import numpy as np
import random
import cv2
from datetime import datetime
from pathlib import Path

# Add project root (parent of gui/) to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # now a Path object

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_CKPT = "/Users/jianuoqiu/Documents/GT/CS8803/MCI-project/models/best_ever/best_model.ckpt"
DB_ROOT = PROJECT_ROOT / "data" / "all"   # where all hdf5 live

from inference_utils.inference_imu2clip import IMU2CLIPInference
from inference_utils.inference_hdf5 import load_hdf5_imu, encode_with_windows


from collection.collect_one_traj import DataCollector
import collection.hdf5_to_csv as hdf5_to_csv
import collection.hdf5_2_mp4 as hdf5_2_mp4
import h5py
from PyQt5.QtWidgets import QListWidgetItem


# GUI Imports

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                             QTreeView, QFileSystemModel, QSlider, QStyle, 
                             QSplitter, QInputDialog, QMessageBox, QListWidget,
                             QFileDialog, QGroupBox, QLineEdit, QSizePolicy, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize, QDir
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# Plotting Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# SIZE CONSTANTS
MAIN_WINDOW_WIDTH = 2200
MAIN_WINDOW_HEIGHT = 800
VIDEO_IMU_SPLIT_SIZE = [500, 400]
QTREE_RIGHT_SPLIT_SIZE = [200, 900]

# --- 1. DUMMY BACKEND ---
class BackendSystem:
    """
    Mock backend to handle data generation, file saving, and querying.
    """
    def generate_dummy_imu_data(self, duration_sec, frequency=50):
        """Generates random 9-axis data (Accel, Gyro, Mag) for a given duration."""
        points = duration_sec * frequency
        time_axis = np.linspace(0, duration_sec, points)
        # 9 axes: 3 Accel, 3 Gyro, 3 Mag
        data = np.random.randn(9, points) 
        return time_axis, data

    def save_to_hdf5(self, data, label):
        """Simulates saving data to an HDF5 file."""
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        print(f"[BACKEND] Saving data with label '{label}' to {filename} (HDF5 format)...")
        # In real implementation: import h5py; with h5py.File(...) as f: ...
        return filename

    def query_database(self, query_sample):
        """Simulates finding matching data in a database."""
        print(f"[BACKEND] Querying database using sample: {query_sample}...")
        time.sleep(0.5) # Simulate network/processing delay
        # Return fake matches
        return [
            "data_walk_01.h5 (98% match)",
            "data_walk_05.h5 (92% match)",
            "data_run_02.h5 (85% match)"
        ]

backend = BackendSystem()

# --- 2. CUSTOM WIDGETS ---

class IMUPlotWidget(QWidget):
    """
    Widget containing Matplotlib canvas with 3 subplots for 9-axis IMU.
    """
    def __init__(self, parent=None, real_time=False):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Create 3 subplots: Accel, Gyro, Mag
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.set_title("Accel", fontsize=10)
        self.ax2 = self.figure.add_subplot(312)
        self.ax2.set_title("Gyro", fontsize=10)
        self.ax3 = self.figure.add_subplot(313)
        self.ax3.set_title("Mag", fontsize=10)
        
        self.axes = [self.ax1, self.ax2, self.ax3]
        self.lines = []
        self.cursor_line = None
        
        # Initialize empty lines
        colors = ['r', 'g', 'b']
        for ax in self.axes:
            ax_lines = []
            for c in colors:
                line, = ax.plot([], [], color=c, linewidth=1)
                ax_lines.append(line)
            self.lines.append(ax_lines)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=8)
        
        # Increase spacing between subplots to avoid overlapping titles
        self.figure.subplots_adjust(hspace=0.35)
        self.figure.tight_layout()

    def plot_static_data(self, t, data):
        """Plots full dataset for Display Mode."""
        t = t - t[0]
        self.t_relative = t

        # data shape: (9, N) -> 0-2 Accel, 3-5 Gyro, 6-8 Mag
        for i in range(3): # For each subplot
            for j in range(3): # For X, Y, Z
                self.lines[i][j].set_data(t, data[i*3 + j])
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        
        # remove old cursor lines
        for l in getattr(self, "cursor_lines", []):
            l.remove()

        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color='k', linestyle='--')
            self.cursor_lines.append(line)
        self.canvas.draw()

    def update_cursor(self, timestamp):
        for line in self.cursor_lines:
            line.set_xdata([timestamp])
        self.canvas.draw_idle()

    # def update_realtime_data(self, t, data):
    #     """Updates plot for Recording Mode."""
    #     # Limit buffer for performance (e.g., last 100 points)
    #     limit = 100
    #     t_view = t[-limit:]
        
    #     for i in range(3):
    #         for j in range(3):
    #             y_view = data[i*3 + j][-limit:]
    #             self.lines[i][j].set_data(t_view, y_view)
            
    #         self.axes[i].set_xlim(min(t_view) if t_view else 0, (max(t_view) + 0.1) if t_view else 1)
    #         if data.shape[1] > 0:
    #             self.axes[i].set_ylim(np.min(data[i*3:i*3+3, -limit:]) - 1, np.max(data[i*3:i*3+3, -limit:]) + 1)
        
    #     self.canvas.draw_idle()

    def update_realtime_data(self, t, data):
        """Updates plot for Recording Mode."""
        # Ensure numpy arrays
        t = np.asarray(t)
        data = np.asarray(data)

        # Nothing to plot
        if t.size == 0 or data.size == 0:
            return

        # Limit buffer for performance (e.g., last 100 points)
        limit = 100
        if t.size < limit:
            limit = t.size

        t_view = t[-limit:]  # shape (limit,)

        for i in range(3):
            # data shape: (9, N) -> [i*3 : i*3+3] is a 3×N block
            for j in range(3):
                y = data[i*3 + j]   # 1D array, length N
                if y.size == 0:
                    continue
                y_view = y[-limit:]
                self.lines[i][j].set_data(t_view, y_view)

            # X axis limits: only if we actually have points
            if t_view.size > 0:
                x_min = float(t_view[0])
                x_max = float(t_view[-1]) + 0.1
                self.axes[i].set_xlim(x_min, x_max)

            # Y axis limits based on last 'limit' samples of 3 channels
            block = data[i*3:i*3+3, -limit:]  # 3×limit
            if block.size > 0:
                y_min = float(np.min(block))
                y_max = float(np.max(block))
                # Avoid zero-height axes
                if y_min == y_max:
                    y_min -= 1.0
                    y_max += 1.0
                self.axes[i].set_ylim(y_min, y_max)

        self.canvas.draw_idle()


# --- 3. MAIN TABS ---

class DisplayModeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        
        # Splitter to resize Sidebar vs Main Content
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # -- Left: File Explorer --
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(QDir.rootPath())
        self.file_model.setNameFilters(["*.hdf5"])
        self.file_model.setNameFilterDisables(False)
        
        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.setRootIndex(self.file_model.index(os.getcwd())) # Start in current dir
        self.tree.setColumnWidth(0, 200)
        # Hide extra columns
        self.tree.setColumnHidden(1, True)  # size
        self.tree.setColumnHidden(2, True)  # type
        self.tree.setColumnHidden(3, True)  # last modified

        self.tree.doubleClicked.connect(self.load_file)
        
        splitter.addWidget(self.tree)
        
        # -- Right: Video + Plot --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Content Splitter (Video vs Plot)
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Video Player
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        
        # 视频用 QLabel 显示 pixmap 帧
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color:black;")
        self.video_label.setMinimumSize(320, 240)
        video_layout.addWidget(self.video_label)

        # 播放控制变量
        self.video_frames = None
        self.video_timestamps = None
        self.current_frame = 0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        # Controls
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.clicked.connect(self.toggle_video)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.slider)
        video_layout.addLayout(controls_layout)
        
        content_splitter.addWidget(video_container)
        
        # IMU Plot
        self.plot_widget = IMUPlotWidget()
        content_splitter.addWidget(self.plot_widget)
        
        # Adjust initial sizes
        content_splitter.setSizes(VIDEO_IMU_SPLIT_SIZE)
        
        right_layout.addWidget(content_splitter)
        splitter.addWidget(right_widget)
        splitter.setSizes(QTREE_RIGHT_SPLIT_SIZE)
        # Dummy data state
        self.duration = 10 # seconds

    def load_file(self, index):
        path = self.file_model.filePath(index)
        if not path.endswith(".hdf5"):
            print("Not an HDF5 file.")
            return
        
        import h5py

        print(f"[INFO] Loading HDF5: {path}")

        with h5py.File(path, "r") as hf:
            self.imu_timestamps = hf["imu/timestamps"][:]
            self.imu_data = hf["imu/data"][:]      # shape N x 9
            self.imu_data = self.imu_data.T
            self.video_timestamps = hf["video/timestamps"][:]
            self.video_frames = hf["video/frames"][:]  # uint8 array (N, H, W, 3)

        print(f"[INFO] Loaded {self.video_frames.shape[0]} video frames")
        print(f"[INFO] Loaded {self.imu_data.shape[1]} IMU samples")

        self.plot_widget.plot_static_data(self.imu_timestamps, self.imu_data)

        # slider range
        self.slider.setRange(0, len(self.video_frames)-1)

        self.current_frame = 0
        self.show_frame(0)

    def show_frame(self, idx):
        frame = self.video_frames[idx]   # H × W × 3

        h, w, c = frame.shape
        image = QImage(frame.data, w, h, 3*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

        # convert absolute timestamp → relative time
        timestamp = self.video_timestamps[idx]
        t0 = self.video_timestamps[0]
        t_relative = (timestamp - t0)
        self.plot_widget.update_cursor(t_relative)


    def toggle_video(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.play_timer.start(33)  # ~30 fps
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame >= len(self.video_frames):
            self.current_frame = 0  # loop
        self.slider.setValue(self.current_frame)
        self.show_frame(self.current_frame)

    def set_position(self, pos):
        self.current_frame = pos
        self.show_frame(pos)

    def on_position_changed(self, position):
        self.slider.setValue(position)
        # Convert ms to seconds for plot
        current_time_sec = position / 1000.0
        self.plot_widget.update_cursor(current_time_sec)

    def on_duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration = duration / 1000.0


class RecordingModeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.collector = None  # will hold DataCollector instance
        self.record_finished_callback = None   # will be set by MainWindow


        # Timer to poll collector and update plot + video
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.update_live_view)

        layout = QHBoxLayout(self)
        
        # -- Left: Camera Feed (Mock) --
        camera_group = QGroupBox("Camera Feed")
        cam_layout = QVBoxLayout(camera_group)
        camera_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.camera_label = QLabel("Camera Feed Placeholder")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setScaledContents(True)
        cam_layout.addWidget(self.camera_label)

        # NEW: status text under the video, so we don’t overwrite the frame
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white; font-size: 12px;")
        cam_layout.addWidget(self.status_label)

        
        # -- Right: Real-time Plot + Controls --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.plot_widget = IMUPlotWidget(real_time=True)
        right_layout.addWidget(self.plot_widget)
        
        # Advanced Recording parameters (hidden by default to avoid squeezing the plot)
        self.params_group = QGroupBox("Recording Params (Advanced)")
        params_layout = QVBoxLayout(self.params_group)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("Folder")); self.input_folder = QLineEdit("test_data"); row1.addWidget(self.input_folder); params_layout.addLayout(row1)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("Task")); self.input_task = QLineEdit("reocrding"); row2.addWidget(self.input_task); params_layout.addLayout(row2)
        row3 = QHBoxLayout(); row3.addWidget(QLabel("Serial Port")); self.input_serial = QLineEdit("/dev/tty.usbmodem101"); row3.addWidget(self.input_serial); params_layout.addLayout(row3)
        row4 = QHBoxLayout(); row4.addWidget(QLabel("Webcam Port")); self.input_webcam = QLineEdit("0"); row4.addWidget(self.input_webcam); params_layout.addLayout(row4)
        row5 = QHBoxLayout(); row5.addWidget(QLabel("Duration (s)")); self.input_duration = QLineEdit("7"); row5.addWidget(self.input_duration); params_layout.addLayout(row5)
        row6 = QHBoxLayout(); row6.addWidget(QLabel("IMU Hz")); self.input_imu = QLineEdit("200"); row6.addWidget(self.input_imu); params_layout.addLayout(row6)
        row7 = QHBoxLayout(); row7.addWidget(QLabel("Camera FPS")); self.input_cam = QLineEdit("30"); row7.addWidget(self.input_cam); params_layout.addLayout(row7)
        task = self.input_task.text().strip()
        if not task:
            task = "recording"
        # Toggle button to show/hide advanced params
        toggle_layout = QHBoxLayout()
        self.btn_toggle_params = QPushButton("Show Params")
        self.btn_toggle_params.setCheckable(True)
        self.btn_toggle_params.toggled.connect(lambda checked: (self.params_group.setVisible(checked), self.btn_toggle_params.setText("Hide Params" if checked else "Show Params")))
        toggle_layout.addWidget(self.btn_toggle_params)
        right_layout.addLayout(toggle_layout)
        
        # Params hidden by default
        self.params_group.setVisible(False)
        right_layout.addWidget(self.params_group)
        
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setStyleSheet("background-color: darkred; color: white; font-weight: bold; padding: 10px;")
        self.btn_record.clicked.connect(self.start_recording)
        right_layout.addWidget(self.btn_record)
        
        layout.addWidget(camera_group, 1)
        layout.addWidget(right_widget, 1)
        
        # Progress timer
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)

        # Timer to wait until data actually starts arriving
        self.wait_timer = QTimer()
        self.wait_timer.timeout.connect(self.check_recording_started)
        
        self.is_recording = False
        self.record_start_time = 0
        self.record_thread = None
        self.record_result = None
        
        # Placeholders for history (kept for plot)
        self.time_history = []
        self.data_history = np.zeros((9, 0))

    def update_live_view(self):
        """Poll DataCollector buffers and update IMU plot + camera preview."""
        if not self.is_recording or self.collector is None:
            return

        collector = self.collector

        # 1) IMU live plot
        with collector.data_lock:
            if len(collector.imu_timestamps) > 0:
                t = np.array(collector.imu_timestamps, dtype=float)
                t = t - t[0]  # relative time
                data = np.array(collector.imu_data, dtype=float)  # shape N x 9

                if data.ndim == 2 and data.shape[0] > 0:
                    data_9xN = data.T  # -> 9 x N
                else:
                    data_9xN = np.zeros((9, 0))

            else:
                t = None
                data_9xN = None

            # 2) last video frame
            if len(collector.video_frames) > 0:
                frame = collector.video_frames[-1].copy()
            else:
                frame = None

        # Update IMU plot
        if t is not None and data_9xN is not None and t.size > 0:
            self.plot_widget.update_realtime_data(t, data_9xN)

        # Update camera preview
        if frame is not None:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.camera_label.setPixmap(
                pixmap.scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )



    def start_recording(self):
        # Read and validate inputs
        folder = self.input_folder.text().strip()
        task = self.input_task.text().strip()
        serial_port = self.input_serial.text().strip()
        try:
            webcam_port = int(self.input_webcam.text().strip())
            duration = int(self.input_duration.text().strip())
            imu_freq = int(self.input_imu.text().strip())
            cam_fps = int(self.input_cam.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric values for webcam port, duration, IMU Hz, and Camera FPS.")
            return
        if not folder or not task or not serial_port:
            QMessageBox.warning(self, "Missing Input", "Folder, Task, and Serial Port are required.")
            return
        
        # Build output path
        output_dir = os.path.join("data", folder, task)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_hdf5 = os.path.join(output_dir, f"{task}_{timestamp}.hdf5")
        
        # UI state
        self.is_recording = True
        self.record_start_time = None  # will be set when data actually arrives
        self.btn_record.setEnabled(False)
        self.btn_record.setText("Recording...")
        self.status_label.setText("Initializing sensors & camera...")

        # 🔁 Instead of starting progress/live timers immediately,
        # wait until DataCollector actually has some data.
        self.wait_timer.start(100)  # check every 100 ms


        # # --- open camera for live preview ---
        # import cv2
        # if self.cap is not None:
        #     self.cap.release()
        #     self.cap = None
        # self.cap = cv2.VideoCapture(webcam_port)
        # if not self.cap.isOpened():
        #     QMessageBox.critical(self, "Camera Error", f"Cannot open webcam {webcam_port}")
        #     return
        # # run preview at ~30 fps
        # self.cam_timer.start(int(1000 / 30))
        
        def worker():
            try:
                # import os
                # import sys

                # # Add project root (parent of gui/) to sys.path
                # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
                # PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
                # if PROJECT_ROOT not in sys.path:
                #     sys.path.insert(0, PROJECT_ROOT)
                # from collection.collect_one_traj import DataCollector
                # import collection.hdf5_to_csv as hdf5_to_csv
                # import collection.hdf5_2_mp4 as hdf5_2_mp4
    
                # def imu_callback(t_array, data_array):
                #     # t_array: 1D list/np.array of timestamps
                #     # data_array: (9, N) array
                #     self.time_history = t_array
                #     self.data_history = data_array
                #     # WARNING: this is called from a thread; in a perfect world use signals.
                #     self.plot_widget.update_realtime_data(self.time_history, self.data_history)

                collector = DataCollector(
                    serial_port=serial_port,
                    baud_rate=115200,
                    webcam_port=webcam_port,
                    imu_freq=imu_freq,
                    camera_freq=cam_fps,
                    output_hdf5=output_hdf5,
                    duration=duration,
                    # realtime_callback=imu_callback   # add this arg
                )
                # expose it to the GUI thread
                self.collector = collector
                ok = collector.start_recording(duration=duration)
                if ok:
                    csv_path = hdf5_to_csv.hdf5_to_csv(output_hdf5)
                    mp4_path = hdf5_2_mp4.hdf5_to_mp4(output_hdf5)
                    self.record_result = ("success", output_hdf5, csv_path, mp4_path)
                else:
                    self.record_result = ("failed",)
            except Exception as e:
                self.record_result = ("error", str(e))
        
        import threading
        self.record_thread = threading.Thread(target=worker, daemon=True)
        self.record_thread.start()

    def check_recording_started(self):
        """Start progress + live view only after some data exists."""
        if not self.is_recording or self.collector is None:
            return

        collector = self.collector

        with collector.data_lock:
            has_imu = len(collector.imu_timestamps) > 0
            has_video = len(collector.video_frames) > 0

        # Wait until we see either IMU OR video data
        if not has_imu and not has_video:
            # still initializing
            return

        # ✅ Data has started coming in → start timers and UI
        self.wait_timer.stop()

        if self.record_start_time is None:
            self.record_start_time = time.time()

        self.progress_timer.start(1000)   # update "X s / Y s" once per second
        self.live_timer.start(100)        # update plots & preview every 100 ms
        self.camera_label.setText("Camera Feed [REC]")


    def update_progress(self):

        # if self.is_recording:
        #     elapsed = time.time() - self.record_start_time
        #     self.camera_label.setText(f"Recording... {elapsed:.0f}s")
        
        # if self.record_thread and not self.record_thread.is_alive() and self.is_recording:
        #     # finalize
        #     self.progress_timer.stop()
        #     self.is_recording = False
        #     self.btn_record.setEnabled(True)
        #     self.btn_record.setText("Start Recording")
        #     self.camera_label.setText("Camera Feed [Idle]")

        if self.is_recording:
            if self.record_start_time is None:
                # Still initializing – timers not started
                self.camera_label.setText("Initializing sensors & camera...")
            else:
                elapsed = time.time() - self.record_start_time
                try:
                    target = int(self.input_duration.text().strip())
                except ValueError:
                    target = 0

                if target and elapsed <= target:
                    self.status_label.setText(f"Recording... {elapsed:.0f}s / {target}s")
                else:
                    # We’re done collecting, now just waiting for file saving / encoding
                    self.status_label.setText("Processing (saving/encoding)...")


        if self.record_thread and not self.record_thread.is_alive() and self.is_recording:
            self.progress_timer.stop()
            self.live_timer.stop()
            self.is_recording = False

            # # stop camera preview
            # if self.cam_timer.isActive():
            #     self.cam_timer.stop()
            # if self.cap is not None:
            #     self.cap.release()
            #     self.cap = None
            self.btn_record.setEnabled(True)
            self.btn_record.setText("Start Recording")
            self.status_label.setText("Camera Feed [Idle]")
            
            res = self.record_result
            if res:
                if res[0] == "success":
                    # Call the callback if set
                    if self.record_finished_callback:
                        self.record_finished_callback(res[1])
                    QMessageBox.information(self, "Recording Saved", f"HDF5: {res[1]}\nCSV: {res[2]}\nMP4: {res[3]}")
                elif res[0] == "failed":
                    QMessageBox.warning(self, "Recording Failed", "Recording failed.")
                elif res[0] == "error":
                    QMessageBox.critical(self, "Error", res[1])


# class QueryModeTab(QWidget):
#     def __init__(self):
#         super().__init__()
#         layout = QVBoxLayout(self)
        
#         # -- Section 1: Select Query Sample --
#         sel_group = QGroupBox("1. Select Query Sample")
#         sel_layout = QHBoxLayout(sel_group)
        
#         self.lbl_selected = QLabel("No file selected")
#         btn_browse = QPushButton("Browse File")
#         btn_browse.clicked.connect(self.browse_file)
        
#         btn_last_rec = QPushButton("Use Last Recording")
#         btn_last_rec.clicked.connect(lambda: self.lbl_selected.setText("Using: Last Recorded Data"))
        
#         sel_layout.addWidget(self.lbl_selected)
#         sel_layout.addWidget(btn_browse)
#         sel_layout.addWidget(btn_last_rec)
        
#         layout.addWidget(sel_group)
        
#         # -- Section 2: Action --
#         self.btn_query = QPushButton("Find Matches in Database")
#         self.btn_query.setFixedHeight(40)
#         self.btn_query.clicked.connect(self.perform_query)
#         layout.addWidget(self.btn_query)
        
#         # -- Section 3: Results --
#         res_group = QGroupBox("3. Matched Results")
#         res_layout = QVBoxLayout(res_group)
#         self.result_list = QListWidget()
#         res_layout.addWidget(self.result_list)
        
#         layout.addWidget(res_group)

#     def browse_file(self):
#         fname, _ = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd(), "HDF5 Files (*.h5)")
#         if fname:
#             self.lbl_selected.setText(f"Using: {os.path.basename(fname)}")

#     def perform_query(self):
#         query_target = self.lbl_selected.text()
#         if "No file" in query_target:
#             QMessageBox.warning(self, "Error", "Please select a sample first.")
#             return

#         self.result_list.clear()
#         self.result_list.addItem("Querying backend...")
#         QApplication.processEvents() # Force UI update
        
#         matches = backend.query_database(query_target)
        
#         self.result_list.clear()
#         for m in matches:
#             self.result_list.addItem(m)

class QueryModeTab(QWidget):
    def __init__(self):
        super().__init__()

        # ----- model -----
        # TODO: adjust device if you want cuda
        self.infer = IMU2CLIPInference(
            checkpoint_path=MODEL_CKPT,
            device="cpu"
        )

        self.query_path = None
        self.video_frames = None
        self.video_timestamps = None
        self.current_frame = 0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        main_layout = QHBoxLayout(self)

        # ========== LEFT: controls + result list ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 1. Select query sample
        sel_group = QGroupBox("1. Select Query Sample")
        sel_layout = QHBoxLayout(sel_group)

        self.lbl_selected = QLabel("No file selected")
        btn_browse = QPushButton("Browse File")
        btn_browse.clicked.connect(self.browse_file)

        btn_last_rec = QPushButton("Use Last Recording")
        btn_last_rec.clicked.connect(self.use_last_recording)

        sel_layout.addWidget(self.lbl_selected)
        sel_layout.addWidget(btn_browse)
        sel_layout.addWidget(btn_last_rec)

        left_layout.addWidget(sel_group)

        # 2. Retrieval settings (checkpoint, DB root, channels, Top K)
        settings_group = QGroupBox("2. Retrieval Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Checkpoint path
        row_ckpt = QHBoxLayout()
        row_ckpt.addWidget(QLabel("Checkpoint"))
        self.input_checkpoint = QLineEdit(str(MODEL_CKPT))
        btn_browse_ckpt = QPushButton("Browse")
        btn_browse_ckpt.clicked.connect(self.browse_checkpoint)
        row_ckpt.addWidget(self.input_checkpoint)
        row_ckpt.addWidget(btn_browse_ckpt)
        settings_layout.addLayout(row_ckpt)

        # Database root
        row_db = QHBoxLayout()
        row_db.addWidget(QLabel("DB Root"))
        self.input_db_root = QLineEdit(str(DB_ROOT))
        btn_browse_db = QPushButton("Browse")
        btn_browse_db.clicked.connect(self.browse_db_root)
        row_db.addWidget(self.input_db_root)
        row_db.addWidget(btn_browse_db)
        settings_layout.addLayout(row_db)

        # Channels + Top K
        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Channels: 21 (fixed)"))
        row_params.addWidget(QLabel("Top K"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 100)
        self.spin_topk.setValue(20)
        row_params.addWidget(self.spin_topk)
        settings_layout.addLayout(row_params)

        left_layout.addWidget(settings_group)

        # Action button
        self.btn_query = QPushButton("Find Matches in Database")
        self.btn_query.setFixedHeight(40)
        self.btn_query.clicked.connect(self.perform_query)
        left_layout.addWidget(self.btn_query)

        # 3. Results list
        res_group = QGroupBox("3. Matched Results")
        res_layout = QVBoxLayout(res_group)
        self.result_list = QListWidget()
        self.result_list.itemDoubleClicked.connect(self.open_match)  # double-click to show
        res_layout.addWidget(self.result_list)
        left_layout.addWidget(res_group)

        main_layout.addWidget(left_widget, 1)

        # ========== RIGHT: video + IMU plot ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # video label
        self.video_label = QLabel("Matched video will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.video_label)

        # [Removed] IMU plot in Query Mode
        # self.plot_widget = IMUPlotWidget()
        # right_layout.addWidget(self.plot_widget)

        main_layout.addWidget(right_widget, 2)

    def browse_checkpoint(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Select checkpoint', str(PROJECT_ROOT), "Checkpoint Files (*.ckpt)"
        )
        if fname:
            self.input_checkpoint.setText(fname)

    def browse_db_root(self):
        dname = QFileDialog.getExistingDirectory(self, 'Select DB root', str(PROJECT_ROOT))
        if dname:
            self.input_db_root.setText(dname)

    def set_last_recording(self, path):
        self.last_recording_path = path

    def use_last_recording(self):
        if hasattr(self, "last_recording_path"):
            self.query_path = self.last_recording_path
            self.lbl_selected.setText(f"Using last recording: {os.path.basename(self.query_path)}")
        else:
            QMessageBox.warning(self, "No recording", "No recording has been made in this session yet.")
    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Open file',
            str(PROJECT_ROOT),
            "HDF5 Files (*.hdf5 *.h5)"
        )
        if fname:
            self.query_path = fname
            self.lbl_selected.setText(f"Using: {os.path.basename(fname)}")

    def perform_query(self):
        if not self.query_path:
            QMessageBox.warning(self, "Error", "Please select a query sample first.")
            return

        # Read retrieval settings (21-channel fixed)
        num_channels = 21
        ckpt_path = self.input_checkpoint.text().strip() or str(MODEL_CKPT)
        db_root = Path(self.input_db_root.text().strip() or str(DB_ROOT))
        top_k = int(self.spin_topk.value())

        # Fixed model selection for 21-channel
        model_name = 'MW2StackRNNPooling21Ch'
        load_channels = 9

        # Instantiate inference model
        try:
            self.infer = IMU2CLIPInference(
                checkpoint_path=ckpt_path,
                device="cpu",
                model_name=model_name,
                num_channels=num_channels,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            return

        # 1. encode query IMU
        try:
            data = load_hdf5_imu(self.query_path, verbose=False, resample_to_200hz=True, num_channels=load_channels)
            imu_data = data['imu_data']
            query_emb = encode_with_windows(self.infer, imu_data)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to encode query:\n{e}")
            return

        # 2. scan database and compute similarities
        self.result_list.clear()
        self.result_list.addItem("Searching database...")
        QApplication.processEvents()

        # Collect HDF5 files under immediate class subfolders, matching CLI behavior
        db_files = []
        if db_root.is_dir():
            for class_dir in sorted([d for d in db_root.iterdir() if d.is_dir()]):
                class_h5s = sorted(class_dir.glob("*.hdf5"))
                # Skip classes with fewer than 1 file? We still include; retrieval is per-query.
                db_files.extend(class_h5s)
        else:
            QMessageBox.warning(self, "No data", f"DB root is not a directory: {db_root}")
            self.result_list.clear()
            return

        if not db_files:
            QMessageBox.warning(self, "No data", f"No HDF5 files found in {db_root}")
            self.result_list.clear()
            return

        matches = []  # list of (sim, path)

        # Exclude the query file itself if it's inside DB root
        query_path_abs = Path(self.query_path).resolve()

        for path in db_files:
            try:
                # Skip self-match
                if Path(path).resolve() == query_path_abs:
                    continue

                # Try to load cached embedding sidecar
                sidecar = Path(str(path) + ".imu2clip.npy")
                if sidecar.exists():
                    db_emb = np.load(sidecar)
                else:
                    data_db = load_hdf5_imu(str(path), verbose=False, resample_to_200hz=True, num_channels=load_channels)
                    db_emb = encode_with_windows(self.infer, data_db['imu_data'])

                # Normalize
                db_emb = db_emb / (np.linalg.norm(db_emb) + 1e-8)

                # cosine similarity (dot product of normalized embeddings)
                sim = float(np.dot(query_emb, db_emb))
                # DEBUG: print per-file similarity
                print(f"[DEBUG] {Path(path).name} sim={sim:.3f} (cached={'Y' if sidecar.exists() else 'N'})")
                if np.isfinite(sim):
                    matches.append((sim, path))
            except Exception as e:
                print(f"[WARN] Failed on {path}: {e}")
                continue

        if not matches:
            self.result_list.clear()
            self.result_list.addItem("No valid matches found.")
            return

        # 3. sort and show top-k
        matches.sort(key=lambda x: x[0], reverse=True)
        top = matches[:top_k]

        # DEBUG: print top-k after sorting
        print("[DEBUG] Top-K after sorting:")
        for rank, (sim, path) in enumerate(top, start=1):
            print(f"  {rank}. {Path(path).name} (sim={sim:.3f})")

        self.result_list.clear()
        for rank, (sim, path) in enumerate(top, start=1):
            item = QListWidgetItem(f"{rank}. {path.name}  (sim={sim:.3f})")
            item.setData(Qt.UserRole, str(path))
            self.result_list.addItem(item)

    def open_match(self, item):
        """Load selected HDF5 and show video + IMU plot."""
        path = item.data(Qt.UserRole)
        if not path:
            return

        try:
            with h5py.File(path, "r") as hf:
                imu_t = hf["imu/timestamps"][:]
                imu_data = hf["imu/data"][:]      # (N, 9) or (N, C)
                imu_data = imu_data.T             # -> (C, N)

                video_t = hf["video/timestamps"][:]
                video_frames = hf["video/frames"][:]  # (N, H, W, 3)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read {path}:\n{e}")
            return

        # update plot
        # self.plot_widget.plot_static_data(imu_t, imu_data)

        # store video
        self.video_frames = video_frames
        self.video_timestamps = video_t
        self.current_frame = 0

        if self.play_timer.isActive():
            self.play_timer.stop()
        self.show_frame(0)
        # Auto-start playback so video and plot cursor advance
        self.play_timer.start(33)

    def show_frame(self, idx):
        if self.video_frames is None or len(self.video_frames) == 0:
            return

        idx = max(0, min(idx, len(self.video_frames) - 1))
        frame = self.video_frames[idx]

        h, w, c = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

        # [Removed] IMU cursor update in Query Mode
        # t0 = self.video_timestamps[0]
        # t_rel = self.video_timestamps[idx] - t0
        # self.plot_widget.update_cursor(t_rel)

    def next_frame(self):
        if self.video_frames is None:
            return
        self.current_frame = (self.current_frame + 1) % len(self.video_frames)
        self.show_frame(self.current_frame)





# --- 4. MAIN WINDOW ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU & Video Analysis Tool")
        self.resize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create Tabs
        self.tab1 = DisplayModeTab()
        self.tab2 = RecordingModeTab()
        self.tab3 = QueryModeTab()

        # connect: when recording finishes successfully, remember path
        self.tab2.record_finished_callback = self.on_record_finished
        
        self.tabs.addTab(self.tab1, "Display Mode")
        self.tabs.addTab(self.tab2, "Recording Mode")
        self.tabs.addTab(self.tab3, "Data Querying")

    def on_record_finished(self, hdf5_path):
        # Called by RecordingModeTab when recording done
        self.tab3.set_last_recording(hdf5_path)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Modern looking style
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()