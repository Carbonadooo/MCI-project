import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D
import h5py 
import sys

# --- CONFIGURATION ---
# ⚠️ EDIT THIS PATH to the specific file you want to test
# TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/high_five_motion/high_five_motion_20251108_210519.hdf5'
# TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/clap_once/clap_once_20251108_191951.hdf5'
# TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/drink_water/drink_water_20251108_210801.hdf5'
# TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/open_close_notebook/open_close_notebook_20251108_211822.hdf5'
# TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/punch_forward/punch_forward_20251108_204358.hdf5'
TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/punch_forward/punch_forward_20251108_204358.hdf5'

# Output directory for the test result
OUTPUT_DIR = '/home/chuye/Documents/MCI-project/data/test_results'
GRAVITY_CONSTANT = 9.81 
# ARROW_SKIP_RATE = 25 
ARROW_SKIP_RATE = 50 
# ---------------------

# --- I. CORE HELPER FUNCTIONS ---

def normalize_vectors(x: pd.Series, y: pd.Series, z: pd.Series, eps: float = 1e-9):
    V = np.vstack([x.values, y.values, z.values]).T.astype(float)
    norms = np.linalg.norm(V, axis=1)
    safe_norms = np.where(norms > eps, norms, np.nan)
    U = V / safe_norms[:, None]
    return U[:, 0], U[:, 1], U[:, 2], norms

def find_imu_data_key(path: Path) -> str:
    with h5py.File(path, 'r') as f:
        available_keys = list(f.keys())
    if 'imu' in available_keys: return 'imu'
    for key in available_keys:
        if 'imu' in key.lower() or 'sensor' in key.lower() or 'data' in key.lower(): return key
    if len(available_keys) == 1: return available_keys[0]
    raise KeyError(f"Could not determine IMU key. Available: {available_keys}")

def read_imu_data(path: Path) -> pd.DataFrame:
    try:
        key = find_imu_data_key(path)
    except Exception as e:
        raise Exception(f"Key detection failed: {e}")
    key = str(key) 
    
    with h5py.File(path, 'r') as f:
        imu_object = f[key]
        
        if isinstance(imu_object, h5py.Group):
            # Check if it has 'data' and 'timestamps' keys (new format)
            if 'data' in imu_object and 'timestamps' in imu_object:
                data = imu_object['data'][()]
                timestamps = imu_object['timestamps'][()]
                
                # Data should be shape (N, 9): [ax, ay, az, mx, my, mz, gx, gy, gz]
                # We need to add timestamps as the first column
                # Convert to DataFrame with timestamp as first column
                full_data = np.column_stack([timestamps, data])
                return pd.DataFrame(full_data)
            else:
                # Find best 2D dataset
                best_name, best_shape = None, (0, 0)
                for name in imu_object.keys():
                    d = imu_object[name]
                    if isinstance(d, h5py.Dataset) and d.ndim == 2:
                        if d.shape[1] > best_shape[1]: 
                            best_name, best_shape = name, d.shape
                if best_name: 
                    data = imu_object[best_name][()]
                    return pd.DataFrame(data)
                else: 
                    raise ValueError("Group found but no 2D dataset inside.")
                    
        elif isinstance(imu_object, h5py.Dataset):
            data = imu_object[()]
            if data.ndim == 2:
                return pd.DataFrame(data)
            else:
                raise ValueError(f"Dataset is not 2D, got shape: {data.shape}")
        else:
            raise TypeError("Unknown HDF5 object type.")

# --- II. RPY CALCULATION ---

def df_to_rpy(df: pd.DataFrame) -> tuple:
    # Clean data first
    df = df.dropna().reset_index(drop=True)
    if df.shape[1] < 10: raise ValueError(f"Expected at least 10 cols (with timestamp), got {df.shape[1]}")
    
    # Columns: [timestamp, ax, ay, az, gx, gy, gz, mx, my, mz]
    # Indices:  [0,        1,  2,  3,  4,  5,  6,  7,  8,  9 ]
    
    acc_unit_x, acc_unit_y, acc_unit_z, acc_norms = normalize_vectors(df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3])
    mag_unit_x, mag_unit_y, mag_unit_z, mag_norms = normalize_vectors(df.iloc[:, 7], df.iloc[:, 8], df.iloc[:, 9])

    # Calculate Roll and Pitch from accelerometer (assuming mostly static or slow motion)
    pitch = np.arcsin(-acc_unit_x) 
    roll = np.arctan2(acc_unit_y, acc_unit_z)

    # Calculate Yaw from magnetometer (compensated for roll and pitch)
    sx, sy, sz = mag_unit_x, mag_unit_y, mag_unit_z
    y_comp = sy * np.cos(roll) + sz * np.sin(roll)
    x_comp = sx * np.cos(pitch) + sy * np.sin(roll) * np.sin(pitch) - sz * np.cos(roll) * np.sin(pitch)
    yaw = np.arctan2(y_comp, x_comp)
    
    return roll, pitch, yaw

def rpy_to_Rot(r, p, y):
    cos_y, sin_y = np.cos(y), np.sin(y)
    cos_p, sin_p = np.cos(p), np.sin(p)
    cos_r, sin_r = np.cos(r), np.sin(r)
    
    N = len(r)
    Rot = np.empty((N, 3, 3))
    Rot[:, 0, 0] = cos_y * cos_p
    Rot[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    Rot[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    Rot[:, 1, 0] = sin_y * cos_p
    Rot[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    Rot[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    Rot[:, 2, 0] = -sin_p
    Rot[:, 2, 1] = cos_p * sin_r
    Rot[:, 2, 2] = cos_p * cos_r
    return Rot

# --- III. TRAJECTORY CALCULATION ---

def calculate_trajectory(
    df: pd.DataFrame,
    rotation_matrices: np.ndarray,
    is_accel_in_g: bool = True,
    gravity: float = 9.81,
) -> np.ndarray:
    """
    Compute 3D trajectory from IMU accelerometer + orientation.

    Assumptions:
      - df columns: [timestamp, ax, ay, az, ...]
      - rotation_matrices[n] is 3x3 body->world rotation for sample n
      - Accelerometer measures proper acceleration (includes gravity)
    """
    # 0. Extract timestamps and build dt (in seconds)
    timestamps = df.iloc[:, 0].values.astype(float)
    if len(timestamps) < 2:
        raise ValueError("Not enough samples to compute trajectory")

    dt = np.diff(timestamps, prepend=timestamps[0])

    # Fix non-positive or zero dt using median positive dt
    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        # Fall back to a default (e.g., 100 Hz)
        median_dt = 0.01
    else:
        median_dt = np.median(positive_dt)
    dt[dt <= 0] = median_dt

    # 1. Get body-frame acceleration and convert units
    accel_raw = df.iloc[:, 1:4].values.astype(float)  # ax, ay, az

    if is_accel_in_g:
        accel_body = accel_raw * gravity  # g -> m/s^2
    else:
        accel_body = accel_raw  # already m/s^2

    # Clean any NaNs
    accel_body = np.nan_to_num(accel_body, nan=0.0)

    # 2. Rotate to world frame: a_world = R * a_body
    if len(rotation_matrices) != len(accel_body):
        raise ValueError(
            f"rotation_matrices length {len(rotation_matrices)} "
            f"does not match accel samples {len(accel_body)}"
        )

    accel_world = np.einsum('nij,nj->ni', rotation_matrices, accel_body)

    # 3. Remove gravity in world frame
    # World Z is up, accelerometer at rest measures +9.81 m/s^2 in +Z,
    # so linear_acc = a_world - [0, 0, 9.81].
    gravity_vec = np.array([0.0, 0.0, gravity], dtype=float)
    accel_linear = accel_world - gravity_vec

    # 3.5. Remove constant bias (VERY important for double integration)
    # For short human motions, mean linear acceleration should be ~0.
    # This step dramatically reduces drift.
    bias = accel_linear.mean(axis=0, keepdims=True)
    accel_linear = accel_linear - bias

    # Final NaN safety
    accel_linear = np.nan_to_num(accel_linear, nan=0.0)

    # 4. First integration: Acc -> Vel (trapezoidal rule)
    vel = np.zeros_like(accel_linear)
    for i in range(1, len(vel)):
        dt_i = dt[i]
        vel[i] = vel[i - 1] + 0.5 * (accel_linear[i] + accel_linear[i - 1]) * dt_i

    # 5. Velocity drift correction (force v_final ≈ 0)
    T_total = timestamps[-1] - timestamps[0]
    if T_total > 1e-6:
        drift_vel = vel[-1] / T_total
        time_normalized = (timestamps - timestamps[0]) / T_total
        vel = vel - np.outer(time_normalized, drift_vel)
    else:
        time_normalized = np.zeros_like(timestamps)

    # 6. Second integration: Vel -> Pos (trapezoidal rule)
    pos = np.zeros_like(vel)
    for i in range(1, len(pos)):
        dt_i = dt[i]
        pos[i] = pos[i - 1] + 0.5 * (vel[i] + vel[i - 1]) * dt_i

    # 7. Position drift correction (force p_final ≈ p_initial = 0)
    if T_total > 1e-6:
        drift_pos = pos[-1] / T_total
        pos = pos - np.outer(time_normalized, drift_pos)

    return pos

# --- IV. VISUALIZATION ---

def visualize_single(trajectory, rotation_matrices, title, save_path):
    if not np.all(np.isfinite(trajectory)):
        print("⚠️ Data contains NaNs. Filtering...")
        mask = np.all(np.isfinite(trajectory), axis=1)
        trajectory = trajectory[mask]
        rotation_matrices = rotation_matrices[mask]
        if len(trajectory) == 0: return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Trajectory (Thicker line for visibility)
    # ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label='Trajectory', color='cyan', linewidth=5)
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label='Trajectory', color='black', linewidth=5)
    
    # Calculate trajectory span
    span = np.ptp(trajectory, axis=0)
    print(f"📏 Trajectory Span: X={span[0]:.3f}m, Y={span[1]:.3f}m, Z={span[2]:.3f}m")
    
    # Calculate Arrow Scale based on the ACTUAL data span
    actual_max_span = span.max()
    arrow_len = actual_max_span * 0.15 if actual_max_span > 0 else 0.01

    # Arrows (show sensor orientation - all three axes)
    step = ARROW_SKIP_RATE
    
    # X-axis (first column of rotation matrix) - RED
    ax.quiver(trajectory[::step,0], trajectory[::step,1], trajectory[::step,2],
              rotation_matrices[::step,0,0], rotation_matrices[::step,1,0], rotation_matrices[::step,2,0],
              length=arrow_len, color='red', normalize=True, alpha=0.7, linewidth=2, label='Sensor X-axis')
    
    # Y-axis (second column of rotation matrix) - GREEN
    ax.quiver(trajectory[::step,0], trajectory[::step,1], trajectory[::step,2],
              rotation_matrices[::step,0,1], rotation_matrices[::step,1,1], rotation_matrices[::step,2,1],
              length=arrow_len, color='green', normalize=True, alpha=0.7, linewidth=2, label='Sensor Y-axis')
    
    # Z-axis (third column of rotation matrix) - BLUE (different from trajectory blue)
    ax.quiver(trajectory[::step,0], trajectory[::step,1], trajectory[::step,2],
              rotation_matrices[::step,0,2], rotation_matrices[::step,1,2], rotation_matrices[::step,2,2],
              length=arrow_len, color='blue', normalize=True, alpha=0.7, linewidth=2, label='Sensor Z-axis')
    
    # Mark start and end points for clarity
    ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], 
               c='lime', marker='o', s=150, edgecolors='black', linewidths=2, label='Start', zorder=10)
    ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], 
               c='orange', marker='X', s=150, edgecolors='black', linewidths=2, label='End', zorder=10)
              
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11) 
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Set equal aspect ratio to avoid distortion
    # Get the center and maximum span
    center = np.mean(trajectory, axis=0)
    max_span = span.max()
    
    # If trajectory is too small, use a minimum range
    plot_range = max(max_span * 0.6, 0.1)  # At least 0.1m range
    
    ax.set_xlim([center[0] - plot_range, center[0] + plot_range])
    ax.set_ylim([center[1] - plot_range, center[1] + plot_range])
    ax.set_zlim([center[2] - plot_range, center[2] + plot_range])
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"💾 Plot saved to: {save_path}")
        
    plt.show()

# --- V. MAIN EXECUTION ---

if __name__ == "__main__":
    target_path = Path(TARGET_FILE)
    
    if not target_path.exists():
        print(f"❌ File not found: {target_path}")
        sys.exit(1)
        
    print(f"🚀 Processing single file: {target_path.name}")
    
    try:
        # 1. Pipeline
        df = read_imu_data(target_path)
        r, p, y = df_to_rpy(df)
        
        # Clean NaNs from RPY and corresponding DataFrame rows
        valid_mask = np.isfinite(r) & np.isfinite(p) & np.isfinite(y)
        if (~valid_mask).sum() > 0:
            print(f"⚠️  Filtering {(~valid_mask).sum()} invalid RPY samples out of {len(r)}")
        
        r, p, y = r[valid_mask], p[valid_mask], y[valid_mask]
        df = df[valid_mask].reset_index(drop=True)
        
        rot = rpy_to_Rot(r, p, y)
        traj = calculate_trajectory(df, rot)
        
        # 2. Visualize
        out_path = Path(OUTPUT_DIR) / target_path.parent.name / (target_path.stem + "_viz.png")
        print(f"📊 Visualizing... (Close plot window to finish)")
        visualize_single(traj, rot, f"Trajectory: {target_path.stem}", out_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        # Print the full traceback to help debug the error
        traceback.print_exc()