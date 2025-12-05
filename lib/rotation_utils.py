"""
Rotation matrix calculation from 9-channel IMU data (accelerometer + gyroscope + magnetometer)

This module provides utilities to compute rotation matrices from IMU sensor readings
and flatten them for use as additional features in the model.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_vector(v):
    """Normalize a vector or array of vectors"""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norms > 1e-9, norms, 1.0)


def get_initial_orientation_from_accel_mag(accel, mag):
    """
    Get initial orientation from accelerometer and magnetometer.
    Assumes device is initially at rest (accel = gravity).
    
    Args:
        accel: (3,) array - accelerometer reading [ax, ay, az]
        mag: (3,) array - magnetometer reading [mx, my, mz]
    
    Returns:
        (3, 3) rotation matrix
    """
    # Normalize accelerometer (should point opposite to gravity in body frame)
    down = -normalize_vector(accel)
    
    # Normalize magnetometer
    mag_norm = normalize_vector(mag)
    
    # Remove mag component along gravity direction
    mag_horizontal = mag_norm - np.dot(mag_norm, down) * down
    mag_horizontal = normalize_vector(mag_horizontal)
    
    # Build rotation matrix
    # Body Z = -down
    # Body Y = mag_horizontal (north direction)
    # Body X = Y × Z (east direction)
    
    z_body = -down
    y_body = mag_horizontal
    x_body = np.cross(z_body, y_body)
    x_body = normalize_vector(x_body)
    
    # Re-orthogonalize
    y_body = np.cross(z_body, x_body)
    y_body = normalize_vector(y_body)
    
    R_init = np.column_stack([x_body, y_body, z_body])
    R_init = R_init.T
    
    return R_init


def integrate_gyroscope_simple(gyro_data, dt=0.005):
    """
    Simple gyroscope integration to get rotation matrix sequence.
    
    Args:
        gyro_data: (N, 3) array of gyroscope data in rad/s [gx, gy, gz]
        dt: time step in seconds (default 0.005 for 200 Hz)
    
    Returns:
        rotation_matrices: (N, 3, 3) array of rotation matrices
    """
    N = len(gyro_data)
    rotation_matrices = np.zeros((N, 3, 3))
    
    # Initialize with identity
    rotation_matrices[0] = np.eye(3)
    
    for i in range(1, N):
        # Current rotation
        R_current = rotation_matrices[i-1]
        
        # Angular velocity in body frame
        omega_body = gyro_data[i]  # [gx, gy, gz] in rad/s
        
        # Rotation increment using Rodrigues formula
        angle = np.linalg.norm(omega_body) * dt
        
        if angle > 1e-6:
            axis = omega_body / np.linalg.norm(omega_body)
            # Use scipy for robust rotation
            r = R.from_rotvec(axis * angle)
            dR = r.as_matrix()
            R_new = R_current @ dR
        else:
            R_new = R_current
        
        rotation_matrices[i] = R_new
    
    return rotation_matrices


def compute_rotation_features(imu_data, sample_rate=200):
    """
    Compute rotation matrix features from 9-channel IMU data.
    
    This function takes a window of IMU data and computes rotation matrices
    using gyroscope integration with accelerometer/magnetometer initialization.
    
    Args:
        imu_data: (9, N) array where:
                  - channels 0-2: accelerometer [ax, ay, az] in m/s² or g
                  - channels 3-5: gyroscope [gx, gy, gz] in deg/s or rad/s
                  - channels 6-8: magnetometer [mx, my, mz] in µT
        sample_rate: sampling rate in Hz (default 200)
    
    Returns:
        rotation_features: (9, N) array - flattened rotation matrices
    """
    assert imu_data.shape[0] == 9, f"Expected 9 channels, got {imu_data.shape[0]}"
    
    N = imu_data.shape[1]
    dt = 1.0 / sample_rate
    
    # Extract sensor data
    accel_data = imu_data[0:3, :].T  # (N, 3)
    gyro_data = imu_data[3:6, :].T   # (N, 3)
    mag_data = imu_data[6:9, :].T    # (N, 3)
    
    # Convert units if needed
    # Assume gyro is in deg/s, convert to rad/s
    # Check if values are likely in deg/s (larger than π)
    if np.abs(gyro_data).max() > np.pi:
        gyro_data = gyro_data * (np.pi / 180)
    
    # Assume accel is in g, convert to m/s²
    # Check if values are small (likely in g)
    if np.abs(accel_data).max() < 20:
        accel_data = accel_data * 9.81
    
    # Initialize rotation matrices
    rotation_matrices = np.zeros((N, 3, 3))
    
    # Get initial orientation from first sample
    try:
        rotation_matrices[0] = get_initial_orientation_from_accel_mag(
            accel_data[0], mag_data[0]
        )
    except:
        # Fallback to identity if initialization fails
        rotation_matrices[0] = np.eye(3)
    
    # Integrate gyroscope
    for i in range(1, N):
        R_current = rotation_matrices[i-1]
        
        # Angular velocity in body frame
        omega_body = gyro_data[i]
        
        # Rotation increment
        angle = np.linalg.norm(omega_body) * dt
        
        if angle > 1e-6:
            axis = omega_body / np.linalg.norm(omega_body)
            # Rotation matrix from axis-angle (Rodrigues formula)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            R_new = R_current @ dR
        else:
            R_new = R_current
        
        rotation_matrices[i] = R_new
    
    # Flatten rotation matrices: each 3x3 matrix becomes 9 values
    # Shape: (N, 3, 3) -> (N, 9) -> (9, N)
    rotation_flat = rotation_matrices.reshape(N, 9).T  # (9, N)
    
    return rotation_flat


def compute_rotation_features_fast(imu_data, sample_rate=200):
    """
    Fast version: compute a single representative rotation matrix for the entire window.
    
    This is faster and may be sufficient for short windows (e.g., 5 seconds).
    
    Args:
        imu_data: (9, N) array of IMU data
        sample_rate: sampling rate in Hz
    
    Returns:
        rotation_features: (9, N) array - repeated flattened rotation matrix
    """
    N = imu_data.shape[1]
    
    # Use middle sample as representative
    mid_idx = N // 2
    
    accel = imu_data[0:3, mid_idx]
    mag = imu_data[6:9, mid_idx]
    
    # Convert accel to m/s² if needed
    if np.abs(accel).max() < 20:
        accel = accel * 9.81
    
    try:
        # Get rotation matrix from middle sample
        R_matrix = get_initial_orientation_from_accel_mag(accel, mag)
    except:
        R_matrix = np.eye(3)
    
    # Flatten to 9 values
    R_flat = R_matrix.flatten()  # (9,)
    
    # Repeat for all time steps
    rotation_features = np.tile(R_flat[:, np.newaxis], (1, N))  # (9, N)
    
    return rotation_features


def compute_world_frame_acceleration(imu_data, rotation_features, sample_rate=200):
    """
    Compute world-frame linear acceleration (gravity-compensated) from IMU data.
    
    This function transforms body-frame acceleration to world frame using rotation
    matrices, then removes the gravity component.
    
    Args:
        imu_data: (9, N) array with raw IMU data (accel, gyro, mag)
        rotation_features: (9, N) array with flattened rotation matrices
                          Each column is [R11, R12, R13, R21, R22, R23, R31, R32, R33]
        sample_rate: sampling rate in Hz (default 200)
        
    Returns:
        world_accel: (3, N) array with world-frame linear acceleration
                     This is acceleration in the world frame with gravity removed
    """
    N = imu_data.shape[1]
    accel_data = imu_data[0:3, :].T  # (N, 3) - body frame acceleration
    
    # Convert units if needed (g → m/s²)
    if np.abs(accel_data).max() < 20:
        accel_data = accel_data * 9.81
    
    # Extract rotation matrices from rotation_features
    # rotation_features is (9, N) where each column is [R11, R12, R13, R21, R22, R23, R31, R32, R33]
    world_accel = np.zeros((3, N))
    
    for i in range(N):
        # Reconstruct 3x3 rotation matrix from flattened form
        R = rotation_features[:, i].reshape(3, 3)
        
        # Transform acceleration to world frame
        accel_world = R @ accel_data[i]
        
        
        # Remove gravity (assuming Z-up world frame: gravity points down = [0, 0, -9.81])
        linear_accel = accel_world + np.array([0, 0, -9.81])
        
        world_accel[:, i] = linear_accel
    
    return world_accel


