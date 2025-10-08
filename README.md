# MCI Project

IMU-driven Query System for Daily Human Manipulation Video Retrieval

| index    | variable         | unit    |
| ---- | ---------- | ----- |
| 1    | timestamp  | s |
| 2-4  | ax, ay, az | g     |
| 5-7  | gx, gy, gz | °/s   |
| 8-10 | mx, my, mz | μT    |


## Preprocess Data

In imu2clip, preprocessing covers both IMU CSVs and raw videos so they can be time-aligned and model-ready.

IMU preprocessing:
Reads raw CSV files, removes invalid timestamps, and sorts them by time. The six IMU channels (accelerometer and gyroscope on x/y/z) are stacked and resampled to 200 Hz using torchaudio.functional.resample().
The resampled signals and new timestamps (in ms) are then saved as .npy files for later use.

Video preprocessing:
Uses FFmpeg to standardize all videos by adjusting frame rate (e.g., 10 fps), cropping the center square, and resizing to a fixed resolution (e.g., 224×224).
The processed clips are saved in a new directory with consistent spatial and temporal resolution.