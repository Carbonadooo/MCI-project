"""
IMU Preprocessing Utilities
---------------------------
This module provides utilities for advanced IMU feature engineering:

    - Orientation estimation from accel+gyro+mag
    - Gyroscope integration for rotation matrices
    - Relative rotation features
    - Gravity-compensated world-frame acceleration
    - Jerk (acceleration derivative)
    - Angular acceleration (gyro derivative)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt


# ===============================
# Vector Normalization
# ===============================

def normalize_vector(v):
    """Normalize a vector or each row of a matrix."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norms > 1e-9, norms, 1.0)


# ===============================
# Initial Orientation from Accel + Mag
# ===============================

def get_initial_orientation_from_accel_mag(accel, mag):
    """
    Estimate initial 3×3 rotation matrix from accel + magnetometer.

    accel: (3,)  raw accel (assumed gravity-dominated)
    mag:   (3,)  raw magnetometer

    Returns:
        R_init: (3, 3) rotation matrix
    """
    # normalized vectors
    down = -normalize_vector(accel)
    mag_norm = normalize_vector(mag)

    # horizontal magnetic component
    mag_horizontal = mag_norm - np.dot(mag_norm, down) * down
    mag_horizontal = normalize_vector(mag_horizontal)

    # Construct basis:
    # Z = -down
    # Y = mag horizontal
    # X = Y × Z
    z_body = -down
    y_body = mag_horizontal
    x_body = normalize_vector(np.cross(z_body, y_body))

    # Re-orthogonalize
    y_body = normalize_vector(np.cross(z_body, x_body))

    R_init = np.column_stack([x_body, y_body, z_body])
    return R_init.T


# ===============================
# Gyroscope Integration
# ===============================

def integrate_gyroscope_simple(gyro_data, dt=0.005):
    """
    Integrate gyroscope readings to produce rotation matrices.

    gyro_data: (N, 3) in rad/s
    dt:        sampling interval

    Returns:
        rotation_matrices: (N, 3, 3)
    """
    N = len(gyro_data)
    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[0] = np.eye(3)

    for i in range(1, N):
        R_prev = rotation_matrices[i-1]
        omega = gyro_data[i]

        angle = np.linalg.norm(omega) * dt

        if angle > 1e-6:
            axis = omega / np.linalg.norm(omega)
            r = R.from_rotvec(axis * angle)
            dR = r.as_matrix()
            R_new = R_prev @ dR
        else:
            R_new = R_prev

        rotation_matrices[i] = R_new

    return rotation_matrices


# ===============================
# Rotation Matrix Features
# ===============================

def compute_rotation_features(imu_data, sample_rate=200):
    """
    imu_data: (9, N)
        accel (0:3), gyro (3:6), mag (6:9)

    Returns:
        rotation_flat: (9, N)
            flattened rotation matrices for each timestep
    """
    assert imu_data.shape[0] == 9, "Expected 9-channel IMU"

    N = imu_data.shape[1]
    dt = 1.0 / sample_rate

    accel = imu_data[0:3].T
    gyro  = imu_data[3:6].T
    mag   = imu_data[6:9].T

    # unit conversion
    if np.abs(gyro).max() > np.pi:
        gyro = gyro * (np.pi / 180.0)  # deg → rad

    if np.abs(accel).max() < 20:
        accel = accel * 9.81  # g → m/s²

    # init rotation
    try:
        R0 = get_initial_orientation_from_accel_mag(accel[0], mag[0])
    except:
        R0 = np.eye(3)

    rotation_matrices = np.zeros((N, 3, 3))
    rotation_matrices[0] = R0

    # integrate gyro
    for i in range(1, N):
        R_prev = rotation_matrices[i-1]
        omega = gyro[i]

        angle = np.linalg.norm(omega) * dt
        if angle > 1e-6:
            axis = omega / np.linalg.norm(omega)
            r = R.from_rotvec(axis * angle)
            dR = r.as_matrix()
            R_new = R_prev @ dR
        else:
            R_new = R_prev

        rotation_matrices[i] = R_new

    # flatten: (N, 3, 3) → (N, 9) → (9, N)
    rotation_flat = rotation_matrices.reshape(N, 9).T
    return rotation_flat


# ===============================
# Gravity-compensated Linear Acceleration
# ===============================

def compute_world_frame_acceleration(imu_data, rotation_features, sample_rate=200):
    """
    imu_data: (9, N)
    rotation_features: (9, N)

    Returns:
        world_accel: (3, N)
    """
    N = imu_data.shape[1]
    accel_data = imu_data[0:3].T  # (N, 3)

    if np.abs(accel_data).max() < 20:
        accel_data = accel_data * 9.81

    world_accel = np.zeros((3, N))

    for i in range(N):
        Rmat = rotation_features[:, i].reshape(3,3)
        accel_world = Rmat @ accel_data[i]

        # remove gravity in world frame (gravity = [0,0,-9.81])
        linear = accel_world + np.array([0,0,-9.81])
        world_accel[:, i] = linear

    return world_accel


# ===============================
# High-level Feature Engineering
# ===============================

def compute_fe_features(imu):
    """
    imu: (T, 9)
    Returns: (T, D)
        - world-frame linear accel (3)
        - jerk (3)
        - angular acceleration (3)
        - relative rotation flatten (9)
        D = 18
    """
    imu_T = imu.T  # (9, T)

    rotation_flat = compute_rotation_features(imu_T)               # (9, T)
    world_accel   = compute_world_frame_acceleration(imu_T, rotation_flat)  # (3, T)

    accel = imu[:, 0:3]
    gyro  = imu[:, 3:6]

    jerk = np.gradient(accel, axis=0)
    ang_acc = np.gradient(gyro, axis=0)

    return np.concatenate([
        world_accel.T,          # (T, 3)
        jerk,                   # (T, 3)
        ang_acc,                # (T, 3)
        rotation_flat.T         # (T, 9)
    ], axis=1)                  # (T, 18)
