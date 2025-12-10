import numpy as np
import matplotlib.pyplot as plt

from imu_ssl.datasets.imu_dataset import build_dataloader
from imu_ssl.datasets.imu_augmentation import (
    Compose, Jitter, Scaling, RandomCrop, TimeWarp, ChannelDropout
)


# ---------------------
# Config
# ---------------------
DATAPATH = "imu_ssl/data"
file_list = [
    f"{DATAPATH}/train_set/pour_water_into_cup/pour_water_into_cup_20251108_210416.hdf5",
    # f"{DATAPATH}/train_set/clap_once/clap_once_20251108_200621.hdf5",
]

# ---------------------
# Define augmentation pipeline
# ---------------------
aug = Compose([
    Jitter(sigma=0.03),
    Scaling(sigma=0.2),
    # RandomCrop(crop_ratio=0.9),
    TimeWarp(sigma=0.3),
    # ChannelDropout(drop_prob=0.05),
])

# ---------------------
# Build dataloader
# ---------------------
dataloader = build_dataloader(
    file_list,
    batch_size=1,              # 用 batch=1 方便可视化
    target_seq_len=1000,
    imu_normalizer=None,
    augment=None
)

# ---------------------
# Get one sample from dataloader
# ---------------------
for batch in dataloader:
    x1, x2 = batch  # Two augmented views
    # Convert to numpy and remove batch dimension
    imu = x1[0].numpy()      # First view (original/augmented)
    imu_aug = x2[0].numpy()  # Second view (augmented)
    break

print(f"Loaded IMU data shapes: {imu.shape}, {imu_aug.shape}")

# ---------------------
# 3 subplots setup (1 for each sensor type)
# ---------------------
sensor_names = ["Accelerometer", "Gyroscope", "Magnetometer"]
axis_labels = ["x", "y", "z"]

# y-axis range for each sensor type
acc_range = (imu[:,0:3].min(), imu[:,0:3].max())
gyro_range = (imu[:,3:6].min(), imu[:,3:6].max())
mag_range = (imu[:,6:9].min(), imu[:,6:9].max())

ranges = [acc_range, gyro_range, mag_range]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for sensor_idx in range(3):  # 3 sensor types
    ax = axes[sensor_idx]
    
    # Plot 3 axes for this sensor type
    start_col = sensor_idx * 3
    for axis_idx in range(3):
        # Plot only view 1 for each axis
        ax.plot(imu[:, start_col + axis_idx], 
               label=f"{axis_labels[axis_idx]}", 
               linewidth=1)
    
    ax.set_title(f"{sensor_names[sensor_idx]}", fontsize=12)
    ax.set_ylim(ranges[sensor_idx])
    ax.legend()
    
    if sensor_idx == 2:  # Bottom subplot
        ax.set_xlabel("Time steps")

plt.tight_layout()
plt.show()




# ---------------------
# Plotting (3 subplots for 3 sensors)
# ---------------------
# fig, axes = plt.subplots(3, 1, figsize=(12, 10))
# axes = axes.ravel()

# # Acc (0,1,2)
# axes[0].plot(imu[:,0], label="orig acc_x")
# axes[0].plot(imu_aug[:,0], label="aug acc_x")
# axes[0].plot(imu[:,1], label="orig acc_y")
# axes[0].plot(imu_aug[:,1], label="aug acc_y")
# axes[0].plot(imu[:,2], label="orig acc_z")
# axes[0].plot(imu_aug[:,2], label="aug acc_z")
# axes[0].set_title("Accelerometer (x,y,z)")
# axes[0].legend()

# # Gyro (3,4,5)
# axes[1].plot(imu[:,3], label="orig gyro_x")
# axes[1].plot(imu_aug[:,3], label="aug gyro_x")
# axes[1].plot(imu[:,4], label="orig gyro_y")
# axes[1].plot(imu_aug[:,4], label="aug gyro_y")
# axes[1].plot(imu[:,5], label="orig gyro_z")
# axes[1].plot(imu_aug[:,5], label="aug gyro_z")
# axes[1].set_title("Gyroscope (x,y,z)")
# axes[1].legend()

# # Mag (6,7,8)
# axes[2].plot(imu[:,6], label="orig mag_x")
# axes[2].plot(imu_aug[:,6], label="aug mag_x")
# axes[2].plot(imu[:,7], label="orig mag_y")
# axes[2].plot(imu_aug[:,7], label="aug mag_y")
# axes[2].plot(imu[:,8], label="orig mag_z")
# axes[2].plot(imu_aug[:,8], label="aug mag_z")
# axes[2].set_title("Magnetometer (x,y,z)")
# axes[2].legend()

# plt.tight_layout()
# plt.show()

