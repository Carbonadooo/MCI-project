#!/usr/bin/env python3
import h5py
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import cv2


class MinimalHDF5Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU & Video Viewer")
        self.root.geometry("1400x800")

        # --- Core data ---
        self.video_frames = None
        self.video_timestamps = None
        self.imu_timestamps = None
        self.imu_data = None

        self.current_frame = 0
        self.is_playing = False
        self.aspect_ratio = None

        # --- Build UI ---
        self.build_ui()

    # ====================== UI ======================
    def build_ui(self):
        # 顶部控制栏
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(top, text="Load HDF5", command=self.load_hdf5_file).pack(side=tk.LEFT)
        self.file_label = ttk.Label(top, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # 控制按钮
        controls = ttk.Frame(top)
        controls.pack(side=tk.RIGHT)
        ttk.Button(controls, text="<<", command=self.prev_frame).pack(side=tk.LEFT, padx=3)
        self.play_button = ttk.Button(controls, text="PLAY", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text=">>", command=self.next_frame).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="REPLAY", command=self.replay_video).pack(side=tk.LEFT, padx=5)
        self.frame_info = ttk.Label(controls, text="Frame: 0/0 | Time: 0.00s")
        self.frame_info.pack(side=tk.LEFT, padx=(10, 0))

        # 主内容：左视频 + 右IMU
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=5)
        content.rowconfigure(0, weight=1)

        # 左视频区域
        video_frame = ttk.LabelFrame(content, text="Video", padding=5)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_area = tk.Frame(video_frame, bg="black")
        self.video_area.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_area, bg="black", text="No video loaded", fg="white")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # 右IMU区域
        imu_frame = ttk.LabelFrame(content, text="IMU Data", padding=5)
        imu_frame.grid(row=0, column=1, sticky="nsew")
        self.create_imu_plot(imu_frame)

    def create_imu_plot(self, parent):
        """Create IMU figure"""
        self.fig = Figure(figsize=(6, 8), dpi=100)
        self.ax_accel = self.fig.add_subplot(311)
        self.ax_gyro = self.fig.add_subplot(312)
        self.ax_mag = self.fig.add_subplot(313)
        for ax, title in zip(
            [self.ax_accel, self.ax_gyro, self.ax_mag],
            ["Accelerometer (m/s²)", "Gyroscope (rad/s)", "Magnetometer (µT)"],
        ):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ====================== HDF5 ======================
    def load_hdf5_file(self):
        path = filedialog.askopenfilename(
            title="Select HDF5 file",
            filetypes=[("HDF5 files", "*.hdf5"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with h5py.File(path, "r") as f:
                self.imu_timestamps = f["imu/timestamps"][:]
                self.imu_data = f["imu/data"][:]
                self.video_timestamps = f["video/timestamps"][:]
                self.video_frames = f["video/frames"][:]
            self.file_label.config(text=f"Loaded: {path.split('/')[-1]}")

            h, w, _ = self.video_frames[0].shape
            self.aspect_ratio = w / h
            self.current_frame = 0
            self.is_playing = False
            self.play_button.config(text="PLAY")

            self.plot_imu_data()
            self.display_current_frame()
            self.update_frame_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    # ====================== VIDEO ======================
    def display_current_frame(self):
        if self.video_frames is None:
            return
        frame = self.video_frames[self.current_frame]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h0, w0, _ = frame_rgb.shape
        aspect = w0 / h0

        area_w, area_h = self.video_area.winfo_width(), self.video_area.winfo_height()
        if area_w <= 1 or area_h <= 1:
            return

        # 保持纵横比缩放
        if area_w / area_h > aspect:
            disp_h = area_h
            disp_w = int(disp_h * aspect)
        else:
            disp_w = area_w
            disp_h = int(disp_w / aspect)

        pil_img = Image.fromarray(frame_rgb).resize((disp_w, disp_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

    # ====================== PLAYBACK ======================
    def toggle_play(self):
        if self.video_frames is None:
            return
        if not self.is_playing:
            self.is_playing = True
            self.play_button.config(text="PAUSE")
            self.play_next_frame()
        else:
            self.is_playing = False
            self.play_button.config(text="PLAY")

    def play_next_frame(self):
        if not self.is_playing or self.video_frames is None:
            return
        self.display_current_frame()
        self.update_frame_info()
        self.current_frame += 1
        if self.current_frame >= len(self.video_frames):
            self.is_playing = False
            self.play_button.config(text="PLAY")
            return
        delay = (self.video_timestamps[self.current_frame] -
                 self.video_timestamps[self.current_frame - 1])
        self.root.after(max(int(delay * 1000), 1), self.play_next_frame)

    def replay_video(self):
        if self.video_frames is None:
            return
        self.is_playing = False
        self.current_frame = 0
        self.play_button.config(text="PLAY")
        self.display_current_frame()
        self.update_frame_info()

    def prev_frame(self):
        if self.video_frames is None:
            return
        self.is_playing = False
        self.current_frame = max(0, self.current_frame - 1)
        self.display_current_frame()
        self.update_frame_info()

    def next_frame(self):
        if self.video_frames is None:
            return
        self.is_playing = False
        self.current_frame = min(len(self.video_frames) - 1, self.current_frame + 1)
        self.display_current_frame()
        self.update_frame_info()

    def update_frame_info(self):
        if self.video_frames is None:
            return
        total = len(self.video_frames)
        current = self.current_frame + 1
        t = self.video_timestamps[self.current_frame] if self.video_timestamps is not None else 0
        self.frame_info.config(text=f"Frame: {current}/{total} | Time: {t:.2f}s")

    # ====================== IMU PLOT ======================
    def plot_imu_data(self):
        if self.imu_data is None:
            return
        self.ax_accel.clear()
        self.ax_gyro.clear()
        self.ax_mag.clear()
        self.ax_accel.plot(self.imu_timestamps, self.imu_data[:, 0:3])
        self.ax_gyro.plot(self.imu_timestamps, self.imu_data[:, 3:6])
        self.ax_mag.plot(self.imu_timestamps, self.imu_data[:, 6:9])
        for ax, title, labels in zip(
            [self.ax_accel, self.ax_gyro, self.ax_mag],
            ["Accelerometer (m/s²)", "Gyroscope (rad/s)", "Magnetometer (µT)"],
            [["ax", "ay", "az"], ["gx", "gy", "gz"], ["mx", "my", "mz"]],
        ):
            ax.set_title(title)
            ax.legend(labels)
            ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = MinimalHDF5Viewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
