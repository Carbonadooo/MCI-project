import sys
import os
import time
import numpy as np
import random
from datetime import datetime

# GUI Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                             QTreeView, QFileSystemModel, QSlider, QStyle, 
                             QSplitter, QInputDialog, QMessageBox, QListWidget,
                             QFileDialog, QGroupBox, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize, QDir
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# Plotting Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        self.ax1.set_title("Accelerometer (X, Y, Z)")
        self.ax2 = self.figure.add_subplot(312)
        self.ax2.set_title("Gyroscope (X, Y, Z)")
        self.ax3 = self.figure.add_subplot(313)
        self.ax3.set_title("Magnetometer (X, Y, Z)")
        
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
        
        self.figure.tight_layout()

    def plot_static_data(self, t, data):
        """Plots full dataset for Display Mode."""
        # data shape: (9, N) -> 0-2 Accel, 3-5 Gyro, 6-8 Mag
        for i in range(3): # For each subplot
            for j in range(3): # For X, Y, Z
                self.lines[i][j].set_data(t, data[i*3 + j])
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        
        # Add vertical cursor
        if self.cursor_line:
            self.cursor_line.remove()
        self.cursor_line = self.ax1.axvline(x=0, color='k', linestyle='--')
        self.canvas.draw()

    def update_cursor(self, timestamp):
        """Updates the vertical line position."""
        if self.cursor_line:
            self.cursor_line.set_xdata([timestamp, timestamp])
            self.canvas.draw_idle()

    def update_realtime_data(self, t, data):
        """Updates plot for Recording Mode."""
        # Limit buffer for performance (e.g., last 100 points)
        limit = 100
        t_view = t[-limit:]
        
        for i in range(3):
            for j in range(3):
                y_view = data[i*3 + j][-limit:]
                self.lines[i][j].set_data(t_view, y_view)
            
            self.axes[i].set_xlim(min(t_view), max(t_view) + 0.1)
            self.axes[i].set_ylim(np.min(data[i*3:i*3+3, -limit:]) - 1, np.max(data[i*3:i*3+3, -limit:]) + 1)
            
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
        
        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.setRootIndex(self.file_model.index(os.getcwd())) # Start in current dir
        self.tree.setColumnWidth(0, 200)
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
        self.video_widget = QVideoWidget()
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        video_layout.addWidget(self.video_widget)
        
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
        content_splitter.setSizes([400, 400])
        
        right_layout.addWidget(content_splitter)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 800])

        # Dummy data state
        self.duration = 10 # seconds

    def load_file(self, index):
        path = self.file_model.filePath(index)
        if path.endswith(('.mp4', '.avi', '.mov')):
            print(f"Loading video: {path}")
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self.btn_play.setEnabled(True)
            
            # Generate fake synchronized IMU data for this file
            t, data = backend.generate_dummy_imu_data(self.duration)
            self.plot_widget.plot_static_data(t, data)
        else:
            print("Selected file is not a supported video format.")

    def toggle_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.media_player.play()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def on_position_changed(self, position):
        self.slider.setValue(position)
        # Convert ms to seconds for plot
        current_time_sec = position / 1000.0
        self.plot_widget.update_cursor(current_time_sec)

    def on_duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration = duration / 1000.0

    def set_position(self, position):
        self.media_player.setPosition(position)


class RecordingModeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        
        # -- Left: Camera Feed (Mock) --
        camera_group = QGroupBox("Camera Feed")
        cam_layout = QVBoxLayout(camera_group)
        self.camera_label = QLabel("Camera Feed Placeholder")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        cam_layout.addWidget(self.camera_label)
        
        # -- Right: Real-time Plot + Controls --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.plot_widget = IMUPlotWidget(real_time=True)
        right_layout.addWidget(self.plot_widget)
        
        self.btn_record = QPushButton("Start Recording (7s)")
        self.btn_record.setStyleSheet("background-color: darkred; color: white; font-weight: bold; padding: 10px;")
        self.btn_record.clicked.connect(self.start_recording)
        right_layout.addWidget(self.btn_record)
        
        layout.addWidget(camera_group, 1)
        layout.addWidget(right_widget, 1)
        
        # Timer for real-time data simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feed)
        self.is_recording = False
        self.record_start_time = 0
        self.recorded_data = [] # Store data here
        self.time_history = []
        self.data_history = np.zeros((9, 0))

    def start_recording(self):
        self.is_recording = True
        self.btn_record.setEnabled(False)
        self.btn_record.setText("Recording...")
        self.time_history = []
        self.data_history = np.zeros((9, 0))
        self.record_start_time = time.time()
        self.timer.start(50) # Update every 50ms

        # Stop after 7 seconds
        QTimer.singleShot(7000, self.stop_recording)

    def update_feed(self):
        # 1. Update Camera (Simulate noise)
        noise = np.random.randint(0, 50)
        self.camera_label.setText(f"Camera Feed [REC]\nFrame: {noise}")
        
        # 2. Update Data
        elapsed = time.time() - self.record_start_time
        new_data = np.random.randn(9, 1) # 1 new sample for 9 axes
        
        self.time_history.append(elapsed)
        self.data_history = np.append(self.data_history, new_data, axis=1)
        
        self.plot_widget.update_realtime_data(self.time_history, self.data_history)

    def stop_recording(self):
        self.timer.stop()
        self.is_recording = False
        self.btn_record.setText("Start Recording (7s)")
        self.btn_record.setEnabled(True)
        self.camera_label.setText("Camera Feed [Idle]")
        
        # Prompt for Label
        text, ok = QInputDialog.getText(self, 'Save Recording', 'Enter activity label:')
        if ok and text:
            backend.save_to_hdf5(self.data_history, text)
            QMessageBox.information(self, "Success", "Recording saved successfully.")


class QueryModeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # -- Section 1: Select Query Sample --
        sel_group = QGroupBox("1. Select Query Sample")
        sel_layout = QHBoxLayout(sel_group)
        
        self.lbl_selected = QLabel("No file selected")
        btn_browse = QPushButton("Browse File")
        btn_browse.clicked.connect(self.browse_file)
        
        btn_last_rec = QPushButton("Use Last Recording")
        btn_last_rec.clicked.connect(lambda: self.lbl_selected.setText("Using: Last Recorded Data"))
        
        sel_layout.addWidget(self.lbl_selected)
        sel_layout.addWidget(btn_browse)
        sel_layout.addWidget(btn_last_rec)
        
        layout.addWidget(sel_group)
        
        # -- Section 2: Action --
        self.btn_query = QPushButton("Find Matches in Database")
        self.btn_query.setFixedHeight(40)
        self.btn_query.clicked.connect(self.perform_query)
        layout.addWidget(self.btn_query)
        
        # -- Section 3: Results --
        res_group = QGroupBox("3. Matched Results")
        res_layout = QVBoxLayout(res_group)
        self.result_list = QListWidget()
        res_layout.addWidget(self.result_list)
        
        layout.addWidget(res_group)

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd(), "HDF5 Files (*.h5)")
        if fname:
            self.lbl_selected.setText(f"Using: {os.path.basename(fname)}")

    def perform_query(self):
        query_target = self.lbl_selected.text()
        if "No file" in query_target:
            QMessageBox.warning(self, "Error", "Please select a sample first.")
            return

        self.result_list.clear()
        self.result_list.addItem("Querying backend...")
        QApplication.processEvents() # Force UI update
        
        matches = backend.query_database(query_target)
        
        self.result_list.clear()
        for m in matches:
            self.result_list.addItem(m)


# --- 4. MAIN WINDOW ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU & Video Analysis Tool")
        self.resize(1200, 800)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create Tabs
        self.tab1 = DisplayModeTab()
        self.tab2 = RecordingModeTab()
        self.tab3 = QueryModeTab()
        
        self.tabs.addTab(self.tab1, "Display Mode")
        self.tabs.addTab(self.tab2, "Recording Mode")
        self.tabs.addTab(self.tab3, "Data Querying")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Modern looking style
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()