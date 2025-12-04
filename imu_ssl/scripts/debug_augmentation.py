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
    f"{DATAPATH}/clap_once/clap_once_20251108_191951.hdf5",
]

# ---------------------
# Build dataloader
# ---------------------
dataloader = build_dataloader(
    file_list,
    batch_size=1,              # 用 batch=1 方便可视化
    target_seq_len=1440,
    imu_normalizer=None,
)

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
# Get one IMU sample
# ---------------------
for batch in dataloader:
    imu = batch["imu"].numpy()[0]   # (T, 9)
    file = batch["file"][0]
    break

print("Loaded:", file)
print("Original shape:", imu.shape)

# ---------------------
# Apply augmentation
# ---------------------
imu_aug = aug(imu.copy())

print("Augmented shape:", imu_aug.shape)


# ---------------------
# 9 subplots setup
# ---------------------
sensor_names = ["acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "mag_x", "mag_y", "mag_z"]

# y-axis grouped by sensor type
acc_range = (
    min(imu[:,0:3].min(), imu_aug[:,0:3].min()),
    max(imu[:,0:3].max(), imu_aug[:,0:3].max()),
)
gyro_range = (
    min(imu[:,3:6].min(), imu_aug[:,3:6].min()),
    max(imu[:,3:6].max(), imu_aug[:,3:6].max()),
)
mag_range = (
    min(imu[:,6:9].min(), imu_aug[:,6:9].min()),
    max(imu[:,6:9].max(), imu_aug[:,6:9].max()),
)


fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.ravel()

for i in range(9):
    ax = axes[i]
    ax.plot(imu[:, i], label="orig", linewidth=1)
    ax.plot(imu_aug[:, i], label="aug", linewidth=1)

    ax.set_title(sensor_names[i], fontsize=10)

    # set y-axis scale for each sensor group
    if i < 3:
        ax.set_ylim(acc_range)
    elif i < 6:
        ax.set_ylim(gyro_range)
    else:
        ax.set_ylim(mag_range)

    if i == 0:
        ax.legend()

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

