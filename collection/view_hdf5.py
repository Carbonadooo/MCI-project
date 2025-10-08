import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import serial
import csv
import os
from collect_one_traj import DataCollector

class HDF5Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU & Video Data Collection & Visualization")
        self.root.geometry("1600x1000")
        
        # Data storage
        self.hdf5_file = None
        self.imu_timestamps = None
        self.imu_data = None
        self.video_timestamps = None
        self.video_frames = None
        self.current_frame_idx = 0
        self.is_playing = False
        self.play_thread = None
        
        # Real-time recording
        self.is_recording = False
        self.data_collector = None
        self.recording_thread = None
        self.realtime_data = {
            'imu_timestamps': [],
            'imu_data': [],
            'video_frames': [],
            'video_timestamps': []
        }
        
        # Default configuration
        self.config = {
            'serial_port': '/dev/ttyACM0',
            'baud_rate': 115200,
            'webcam_port': 0,
            'imu_frequency': 200,
            'camera_frequency': 30,
            'recording_duration': 15
        }

        self.display_hz = 60.0  # UI refresh rate for playback (try 60 or 120)
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - File operations
        left_controls = ttk.Frame(top_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(left_controls, text="Load HDF5 File", command=self.load_hdf5_file).pack(side=tk.LEFT, padx=(0, 10))
        self.file_label = ttk.Label(left_controls, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Center - Recording controls
        center_controls = ttk.Frame(top_frame)
        center_controls.pack(side=tk.LEFT, padx=20)
        
        self.record_button = ttk.Button(center_controls, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(center_controls, text="Settings", command=self.open_settings).pack(side=tk.LEFT, padx=(0, 10))
        
        # Right side - Video controls
        video_control_frame = ttk.Frame(top_frame)
        video_control_frame.pack(side=tk.RIGHT)
        
        ttk.Button(video_control_frame, text="⏮", command=self.first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_control_frame, text="⏪", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(video_control_frame, text="▶", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(video_control_frame, text="⏩", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_control_frame, text="⏭", command=self.last_frame).pack(side=tk.LEFT, padx=2)
        
        # Frame info
        self.frame_info_label = ttk.Label(video_control_frame, text="Frame: 0/0")
        self.frame_info_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for video
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video display
        video_frame = ttk.LabelFrame(left_panel, text="Video Stream", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame, text="Load HDF5 file to view video", 
                                   background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Video timeline
        timeline_frame = ttk.Frame(video_frame)
        timeline_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.timeline_var = tk.DoubleVar()
        self.timeline_scale = ttk.Scale(timeline_frame, from_=0, to=100, 
                                      variable=self.timeline_var, orient=tk.HORIZONTAL,
                                      command=self.on_timeline_change)
        self.timeline_scale.pack(fill=tk.X)
        
        # Right panel for IMU plots
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create notebook for IMU plots
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for different sensor groups
        self.create_imu_tabs()
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def open_settings(self):
        """Open settings dialog for recording configuration"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Recording Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # Make window modal
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Center the window
        settings_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Create settings form
        main_frame = ttk.Frame(settings_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Serial port settings
        ttk.Label(main_frame, text="Serial Port:").grid(row=0, column=0, sticky=tk.W, pady=5)
        serial_var = tk.StringVar(value=self.config['serial_port'])
        serial_entry = ttk.Entry(main_frame, textvariable=serial_var, width=20)
        serial_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Baud rate
        ttk.Label(main_frame, text="Baud Rate:").grid(row=1, column=0, sticky=tk.W, pady=5)
        baud_var = tk.StringVar(value=str(self.config['baud_rate']))
        baud_combo = ttk.Combobox(main_frame, textvariable=baud_var, width=17, state="readonly")
        baud_combo['values'] = ('9600', '19200', '38400', '57600', '115200', '230400', '460800', '921600')
        baud_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Webcam port
        ttk.Label(main_frame, text="Webcam Port:").grid(row=2, column=0, sticky=tk.W, pady=5)
        webcam_var = tk.StringVar(value=str(self.config['webcam_port']))
        webcam_entry = ttk.Entry(main_frame, textvariable=webcam_var, width=20)
        webcam_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # IMU frequency
        ttk.Label(main_frame, text="IMU Frequency (Hz):").grid(row=3, column=0, sticky=tk.W, pady=5)
        imu_freq_var = tk.StringVar(value=str(self.config['imu_frequency']))
        imu_freq_combo = ttk.Combobox(main_frame, textvariable=imu_freq_var, width=17, state="readonly")
        imu_freq_combo['values'] = ('50', '100', '200', '250', '500', '1000')
        imu_freq_combo.grid(row=3, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Camera frequency
        ttk.Label(main_frame, text="Camera Frequency (FPS):").grid(row=4, column=0, sticky=tk.W, pady=5)
        cam_freq_var = tk.StringVar(value=str(self.config['camera_frequency']))
        cam_freq_combo = ttk.Combobox(main_frame, textvariable=cam_freq_var, width=17, state="readonly")
        cam_freq_combo['values'] = ('15', '24', '30', '60', '120')
        cam_freq_combo.grid(row=4, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Recording duration
        ttk.Label(main_frame, text="Duration (seconds):").grid(row=5, column=0, sticky=tk.W, pady=5)
        duration_var = tk.StringVar(value=str(self.config['recording_duration']))
        duration_entry = ttk.Entry(main_frame, textvariable=duration_var, width=20)
        duration_entry.grid(row=5, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        def save_settings():
            try:
                self.config['serial_port'] = serial_var.get()
                self.config['baud_rate'] = int(baud_var.get())
                self.config['webcam_port'] = int(webcam_var.get())
                self.config['imu_frequency'] = int(imu_freq_var.get())
                self.config['camera_frequency'] = int(cam_freq_var.get())
                self.config['recording_duration'] = int(duration_var.get())
                settings_window.destroy()
                self.status_label.config(text="Settings updated")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid setting value: {e}")
        
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def toggle_recording(self):
        """Toggle real-time recording"""
        if not self.is_recording:
            self.start_realtime_recording()
        else:
            self.stop_realtime_recording()
    
    def start_realtime_recording(self):
        """Start real-time recording and visualization"""
        try:
            self.status_label.config(text="Initializing recording...")
            self.root.update()
            
            # Create data collector with current settings
            self.data_collector = DataCollector(
                serial_port=self.config['serial_port'],
                baud_rate=self.config['baud_rate'],
                webcam_port=self.config['webcam_port'],
                imu_freq=self.config['imu_frequency'],
                camera_freq=self.config['camera_frequency']
            )
            
            # Initialize hardware
            if not self.data_collector.initialize_hardware():
                messagebox.showerror("Error", "Failed to initialize hardware. Check connections and settings.")
                return
            
            # Start recording in a separate thread
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.recording_thread = threading.Thread(target=self.realtime_recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Start real-time visualization
            self.start_realtime_visualization()
            
            self.status_label.config(text="Recording in progress...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.is_recording = False
            self.record_button.config(text="Start Recording")
    
    def stop_realtime_recording(self):
        """Stop real-time recording"""
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        
        if self.data_collector:
            self.data_collector.recording = False
            self.data_collector.cleanup_hardware()
        
        self.status_label.config(text="Recording stopped")
    
    def realtime_recording_loop(self):
        """Real-time recording loop"""
        try:
            # Start the data collector
            self.data_collector.recording = True
            self.data_collector.start_time = time.time()
            
            # Create threads for data collection
            imu_thread = threading.Thread(target=self.data_collector.record_imu_data)
            video_thread = threading.Thread(target=self.data_collector.record_video)
            
            imu_thread.start()
            video_thread.start()
            
            # Monitor recording duration
            start_time = time.time()
            duration = self.config['recording_duration']
            
            while self.is_recording and (time.time() - start_time) < duration:
                time.sleep(0.1)
            
            # Stop recording
            self.data_collector.recording = False
            
            # Wait for threads to finish
            imu_thread.join()
            video_thread.join()
            
            # Copy data to realtime storage
            with self.data_collector.data_lock:
                self.realtime_data['imu_timestamps'] = self.data_collector.imu_timestamps.copy()
                self.realtime_data['imu_data'] = self.data_collector.imu_data.copy()
                self.realtime_data['video_frames'] = self.data_collector.video_frames.copy()
                self.realtime_data['video_timestamps'] = self.data_collector.video_timestamps.copy()
            
            # Update display with recorded data
            self.root.after(0, self.load_realtime_data)
            
        except Exception as e:
            print(f"Recording error: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"Recording error: {e}"))
    
    def start_realtime_visualization(self):
        """Start real-time visualization during recording"""
        # This will be called periodically to update the display
        if self.is_recording:
            self.update_realtime_display()
            self.root.after(100, self.start_realtime_visualization)  # Update every 100ms
    
    def update_realtime_display(self):
        """Update display with real-time data"""
        if not self.is_recording or not self.data_collector:
            return
        
        try:
            # Get latest data
            with self.data_collector.data_lock:
                if len(self.data_collector.imu_timestamps) > 0:
                    # Update IMU plots with latest data
                    self.update_realtime_imu_plots()
                
                if len(self.data_collector.video_frames) > 0:
                    # Update video display with latest frame
                    self.display_latest_frame()
        
        except Exception as e:
            print(f"Real-time display error: {e}")
    
    def display_latest_frame(self):
        """Display the latest video frame"""
        if not self.data_collector or len(self.data_collector.video_frames) == 0:
            return
        
        try:
            # Get the latest frame
            latest_frame = self.data_collector.video_frames[-1]
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit display
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 600, 400
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Frame display error: {e}")
    
    def update_realtime_imu_plots(self):
        """Update IMU plots with real-time data"""
        if not self.data_collector or len(self.data_collector.imu_timestamps) == 0:
            return
        
        try:
            # Convert to numpy arrays for plotting
            timestamps = np.array(self.data_collector.imu_timestamps)
            data = np.array(self.data_collector.imu_data)
            
            # Update individual sensor group plots
            self.update_realtime_sensor_plot("accelerometer", [0, 1, 2], timestamps, data)
            self.update_realtime_sensor_plot("gyroscope", [3, 4, 5], timestamps, data)
            self.update_realtime_sensor_plot("magnetometer", [6, 7, 8], timestamps, data)
            
        except Exception as e:
            print(f"Real-time IMU plot error: {e}")
    
    def update_realtime_sensor_plot(self, group_name, data_indices, timestamps, data):
        """Update a specific sensor group plot with real-time data"""
        try:
            fig = getattr(self, f"{group_name}_fig")
            ax = getattr(self, f"{group_name}_ax")
            canvas = getattr(self, f"{group_name}_canvas")
            labels = getattr(self, f"{group_name}_labels")
            colors = getattr(self, f"{group_name}_colors")
            
            # Clear previous plots
            ax.clear()
            
            # Plot each sensor
            for i, (idx, label, color) in enumerate(zip(data_indices, labels, colors)):
                ax.plot(timestamps, data[:, idx], color=color, label=label, linewidth=1.5)
            
            # Configure plot
            ax.set_title(f"{group_name.title()} Data (Real-time)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_ylabel(f"{group_name.title()} ({getattr(self, f'{group_name}_units')})", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            canvas.draw()
            
        except Exception as e:
            print(f"Real-time sensor plot error: {e}")
    
    def load_realtime_data(self):
        """Load recorded real-time data into the viewer"""
        try:
            # Copy real-time data to main data storage
            self.imu_timestamps = np.array(self.realtime_data['imu_timestamps'])
            self.imu_data = np.array(self.realtime_data['imu_data'])
            self.video_timestamps = np.array(self.realtime_data['video_timestamps'])
            self.video_frames = np.array(self.realtime_data['video_frames'])
            
            # Update video controls
            self.current_frame_idx = 0
            self.timeline_scale.config(to=len(self.video_frames) - 1)
            self.update_frame_info()
            
            # Update video display
            self.display_current_frame()
            
            # Update IMU plots
            self.update_imu_plots()
            
            self.status_label.config(text=f"Loaded: {len(self.imu_timestamps)} IMU samples, {len(self.video_frames)} video frames")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load real-time data: {e}")
            self.status_label.config(text="Error loading real-time data")
        
    def create_imu_tabs(self):
        """Create tabs for different IMU sensor groups"""
        # Accelerometer tab
        accel_frame = ttk.Frame(self.notebook)
        self.notebook.add(accel_frame, text="Accelerometer")
        self.create_imu_plot(accel_frame, "Accelerometer", ["ax", "ay", "az"], 
                           ["red", "green", "blue"], "m/s²")
        
        # Gyroscope tab
        gyro_frame = ttk.Frame(self.notebook)
        self.notebook.add(gyro_frame, text="Gyroscope")
        self.create_imu_plot(gyro_frame, "Gyroscope", ["gx", "gy", "gz"], 
                           ["red", "green", "blue"], "rad/s")
        
        # Magnetometer tab
        mag_frame = ttk.Frame(self.notebook)
        self.notebook.add(mag_frame, text="Magnetometer")
        self.create_imu_plot(mag_frame, "Magnetometer", ["mx", "my", "mz"], 
                           ["red", "green", "blue"], "µT")
        
        # All sensors tab
        all_frame = ttk.Frame(self.notebook)
        self.notebook.add(all_frame, text="All Sensors")
        self.create_all_sensors_plot(all_frame)
        
    def create_imu_plot(self, parent, title, labels, colors, units):
        """Create IMU plot for a specific sensor group"""
        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Configure plot
        ax.set_title(f"{title} Data", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel(f"{title} ({units})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store references
        setattr(self, f"{title.lower()}_fig", fig)
        setattr(self, f"{title.lower()}_ax", ax)
        setattr(self, f"{title.lower()}_canvas", canvas)
        setattr(self, f"{title.lower()}_labels", labels)
        setattr(self, f"{title.lower()}_colors", colors)
        setattr(self, f"{title.lower()}_units", units)
        
    def create_all_sensors_plot(self, parent):
        """Create plot showing all 9 sensors"""
        fig = Figure(figsize=(8, 12), dpi=100)
        
        # Create 3 subplots for each sensor group
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
        # Configure subplots
        axes = [ax1, ax2, ax3]
        titles = ["Accelerometer (m/s²)", "Gyroscope (rad/s)", "Magnetometer (µT)"]
        labels_groups = [["ax", "ay", "az"], ["gx", "gy", "gz"], ["mx", "my", "mz"]]
        colors = ["red", "green", "blue"]
        
        for i, (ax, title, labels) in enumerate(zip(axes, titles, labels_groups)):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel(title.split()[0], fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store references
        self.all_fig = fig
        self.all_axes = axes
        self.all_canvas = canvas
        self.all_labels_groups = labels_groups
        self.all_colors = colors
        
    def load_hdf5_file(self):
        """Load HDF5 file and extract data"""
        file_path = filedialog.askopenfilename(
            title="Select HDF5 file",
            filetypes=[("HDF5 files", "*.hdf5"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_label.config(text="Loading HDF5 file...")
            self.root.update()
            
            # Open HDF5 file
            with h5py.File(file_path, 'r') as hf:
                # Load IMU data
                self.imu_timestamps = hf['imu/timestamps'][:]
                self.imu_data = hf['imu/data'][:]
                
                # Load video data
                self.video_timestamps = hf['video/timestamps'][:]
                self.video_frames = hf['video/frames'][:]
                
                # Get metadata
                self.metadata = dict(hf.attrs)
                
            # Update file label
            self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            
            # Update video controls
            self.current_frame_idx = 0
            self.timeline_scale.config(to=len(self.video_frames) - 1)
            self.update_frame_info()
            
            # Update video display
            self.display_current_frame()
            
            # Update IMU plots
            self.update_imu_plots()
            
            self.status_label.config(text=f"Loaded: {len(self.imu_timestamps)} IMU samples, {len(self.video_frames)} video frames")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load HDF5 file:\n{str(e)}")
            self.status_label.config(text="Error loading file")
            
    def display_current_frame(self):
        """Display the current video frame"""
        if self.video_frames is None:
            return
            
        # Get current frame
        frame = self.video_frames[self.current_frame_idx]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit display
        height, width = frame_rgb.shape[:2]
        max_width, max_height = 600, 400
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # Keep a reference
        
        # Update timeline
        self.timeline_var.set(self.current_frame_idx)
        
    def update_frame_info(self):
        """Update frame information display"""
        if self.video_frames is not None:
            total_frames = len(self.video_frames)
            current_time = self.video_timestamps[self.current_frame_idx] if self.video_timestamps is not None else 0
            self.frame_info_label.config(text=f"Frame: {self.current_frame_idx + 1}/{total_frames} | Time: {current_time:.2f}s")
            
    def update_imu_plots(self):
        """Update all IMU plots with loaded data"""
        if self.imu_data is None or self.imu_timestamps is None:
            return
            
        # Update individual sensor group plots
        self.update_sensor_group_plot("accelerometer", [0, 1, 2])
        self.update_sensor_group_plot("gyroscope", [3, 4, 5])
        self.update_sensor_group_plot("magnetometer", [6, 7, 8])
        
        # Update all sensors plot
        self.update_all_sensors_plot()
        
    def update_sensor_group_plot(self, group_name, data_indices):
        """Update a specific sensor group plot"""
        fig = getattr(self, f"{group_name}_fig")
        ax = getattr(self, f"{group_name}_ax")
        canvas = getattr(self, f"{group_name}_canvas")
        labels = getattr(self, f"{group_name}_labels")
        colors = getattr(self, f"{group_name}_colors")
        
        # Clear previous plots
        ax.clear()
        
        # Plot each sensor
        for i, (idx, label, color) in enumerate(zip(data_indices, labels, colors)):
            ax.plot(self.imu_timestamps, self.imu_data[:, idx], 
                   color=color, label=label, linewidth=1.5)
        
        # Configure plot
        ax.set_title(f"{group_name.title()} Data", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel(f"{group_name.title()} ({getattr(self, f'{group_name}_units')})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add current time indicator
        if self.video_timestamps is not None and self.current_frame_idx < len(self.video_timestamps):
            current_time = self.video_timestamps[self.current_frame_idx]
            ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        canvas.draw()
        
    def update_all_sensors_plot(self):
        """Update the all sensors plot"""
        if not hasattr(self, 'all_fig'):
            return
            
        # Clear all subplots
        for ax in self.all_axes:
            ax.clear()
        
        # Plot each sensor group
        data_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        titles = ["Accelerometer (m/s²)", "Gyroscope (rad/s)", "Magnetometer (µT)"]
        
        for ax, indices, title, labels in zip(self.all_axes, data_indices, titles, self.all_labels_groups):
            for i, (idx, label, color) in enumerate(zip(indices, labels, self.all_colors)):
                ax.plot(self.imu_timestamps, self.imu_data[:, idx], 
                       color=color, label=label, linewidth=1.5)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel(title.split()[0], fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add current time indicator
            if self.video_timestamps is not None and self.current_frame_idx < len(self.video_timestamps):
                current_time = self.video_timestamps[self.current_frame_idx]
                ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        self.all_fig.tight_layout()
        self.all_canvas.draw()
        
    def on_timeline_change(self, value):
        """Handle timeline slider change"""
        if self.video_frames is not None:
            self.current_frame_idx = int(float(value))
            self.display_current_frame()
            self.update_frame_info()
            self.update_imu_plots()
            
    def first_frame(self):
        """Go to first frame"""
        if self.video_frames is not None:
            self.current_frame_idx = 0
            self.display_current_frame()
            self.update_frame_info()
            self.update_imu_plots()
            
    def last_frame(self):
        """Go to last frame"""
        if self.video_frames is not None:
            self.current_frame_idx = len(self.video_frames) - 1
            self.display_current_frame()
            self.update_frame_info()
            self.update_imu_plots()
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.video_frames is not None and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_current_frame()
            self.update_frame_info()
            self.update_imu_plots()
            
    def next_frame(self):
        """Go to next frame"""
        if self.video_frames is not None and self.current_frame_idx < len(self.video_frames) - 1:
            self.current_frame_idx += 1
            self.display_current_frame()
            self.update_frame_info()
            self.update_imu_plots()
            
    # def toggle_play(self):
    #     """Toggle playback on/off"""
    #     if self.video_frames is None or len(self.video_frames) == 0:
    #         return

    #     # Toggle playback state
    #     if self.is_playing:
    #         # Stop playback
    #         self.is_playing = False
    #         self.play_button.config(text="▶")
    #     else:
    #         # Start playback in real time
    #         self.is_playing = True
    #         self.play_button.config(text="⏸")
    #         self.play_thread = threading.Thread(target=self.play_video_real_time)
    #         self.play_thread.daemon = True
    #         self.play_thread.start()
    def toggle_play(self):
        """Toggle playback on/off (real-time with high-frequency display)."""
        if self.video_frames is None or len(self.video_frames) == 0:
            return

        if self.is_playing:
            self.is_playing = False
            self.play_button.config(text="▶")
            return

        self.is_playing = True
        self.play_button.config(text="⏸")

        # Precompute normalized timestamps
        self._video_ts0 = self.video_timestamps - self.video_timestamps[0]
        self._imu_ts0 = (self.imu_timestamps - self.imu_timestamps[0]) if self.imu_timestamps is not None else None

        th = threading.Thread(target=self.play_video_realtime_high_refresh, daemon=True)
        self.play_thread = th
        th.start()



    def play_video_real_time(self):
        """Play video and IMU data synchronized to real recorded speed."""
        if self.video_timestamps is None or len(self.video_timestamps) < 2:
            return

        # Normalize timestamps so they start at 0
        video_ts = self.video_timestamps - self.video_timestamps[0]
        imu_ts = self.imu_timestamps - self.imu_timestamps[0] if self.imu_timestamps is not None else None

        # Start playback from the current frame
        start_idx = self.current_frame_idx
        start_time = time.perf_counter()
        t0_video = video_ts[start_idx]  # recorded time offset at current frame

        while self.is_playing and self.current_frame_idx < len(video_ts) - 1:
            i = self.current_frame_idx
            current_play_time = time.perf_counter() - start_time + t0_video

            # Find the next frame to display based on elapsed real time
            while (self.current_frame_idx < len(video_ts) - 1 and
                video_ts[self.current_frame_idx + 1] <= current_play_time):
                self.current_frame_idx += 1

            # Update the GUI (video + IMU plots)
            self.root.after(0, self.display_current_frame)
            self.root.after(0, self.update_frame_info)
            self.root.after(0, self.update_imu_plots)

            # Compute time until next frame (to maintain real speed)
            if self.current_frame_idx < len(video_ts) - 1:
                next_time = video_ts[self.current_frame_idx + 1]
                delay = next_time - current_play_time
                if delay > 0:
                    time.sleep(delay)
            else:
                break

        # When playback ends
        self.is_playing = False
        self.root.after(0, lambda: self.play_button.config(text="▶"))


    def play_video_realtime_high_refresh(self):
        """
        Real-time playback with high-frequency UI updates.
        - Keeps wall-clock speed equal to the recorded timeline.
        - Renders at self.display_hz (e.g., 60 Hz) and holds the last video frame between recorded frames.
        - IMU plots are updated at the same high refresh so the time cursor moves smoothly.
        """
        video_ts = self._video_ts0
        if video_ts is None or len(video_ts) == 0:
            self.is_playing = False
            self.root.after(0, lambda: self.play_button.config(text="▶"))
            return

        # Playback clock setup
        tick_dt = 1.0 / max(1.0, float(self.display_hz))
        start_idx = min(self.current_frame_idx, len(video_ts) - 1)
        t_media_start = float(video_ts[start_idx])      # media time at playback start
        t_media_end = float(video_ts[-1])               # last media time
        clk0 = time.perf_counter()                      # wall-clock reference
        next_tick_wall = clk0                           # next UI update at 60 Hz
        last_idx = -1

        while self.is_playing:
            # Current wall-clock elapsed since we started playing
            now = time.perf_counter()
            t_media = t_media_start + (now - clk0)      # target media time (real-time)

            # Stop when media time passes the recording end
            if t_media > t_media_end:
                break

            # Choose the last frame with ts <= t_media (hold-last between frames)
            idx = np.searchsorted(video_ts, t_media, side='right') - 1
            if idx < 0:
                idx = 0
            if idx >= len(self.video_frames):
                idx = len(self.video_frames) - 1

            # Only trigger GUI updates if frame index changed or it’s time for a tick redraw
            if idx != last_idx:
                self.current_frame_idx = idx
                last_idx = idx
                self.root.after(0, self.display_current_frame)
                self.root.after(0, self.update_frame_info)

            # Always move the IMU time cursor at the display rate for smoothness
            self.root.after(0, self.update_imu_plots)

            # Compute when to wake up next:
            #   - regular UI tick (60 Hz), and
            #   - the exact time of the next recorded video frame (to reduce latency)
            next_tick_wall += tick_dt

            # Next recorded frame's media time (if any)
            if idx < len(video_ts) - 1:
                next_frame_media = float(video_ts[idx + 1])
                # Convert that media time to wall clock
                next_frame_wall = clk0 + (next_frame_media - t_media_start)
                # Wake up at the EARLIER of (next 60 Hz tick) or (next frame boundary)
                wake_at = min(next_tick_wall, next_frame_wall)
            else:
                # No next frame; just tick at display rate
                wake_at = next_tick_wall

            # Sleep until wake_at (but not negative)
            remaining = wake_at - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # If we're late (e.g., heavy GUI), resync the 60 Hz tick to avoid drift
                next_tick_wall = time.perf_counter()

        # Done
        self.is_playing = False
        self.root.after(0, lambda: self.play_button.config(text="▶"))


            
    # def play_video(self):
    #     """Play video in a separate thread"""
    #     while self.is_playing and self.video_frames is not None:
    #         if self.current_frame_idx < len(self.video_frames) - 1:
    #             self.current_frame_idx += 1
    #             # Update GUI in main thread
    #             self.root.after(0, self.display_current_frame)
    #             self.root.after(0, self.update_frame_info)
    #             self.root.after(0, self.update_imu_plots)
    #             time.sleep(1/30)  # 30 FPS playback
    #         else:
    #             self.is_playing = False
    #             self.root.after(0, lambda: self.play_button.config(text="▶"))

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = HDF5Viewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
