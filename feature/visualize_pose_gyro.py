import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import h5py
import sys
from PIL import Image
import io

# --- CONFIGURATION ---
TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/pick_and_place/pick_and_place_20251108_202340.hdf5'
OUTPUT_DIR = '/home/chuye/Documents/MCI-project/data/pose_animations'
FPS = 30  # Frames per second for the GIF
ARROW_LENGTH = 0.5  # Length of orientation arrows
FRAME_SKIP = 5  # Show every Nth frame (to speed up animation)
# ---------------------

# --- HELPER FUNCTIONS ---

def find_imu_data_key(path: Path) -> str:
    with h5py.File(path, 'r') as f:
        available_keys = list(f.keys())
    if 'imu' in available_keys: return 'imu'
    for key in available_keys:
        if 'imu' in key.lower() or 'sensor' in key.lower() or 'data' in key.lower(): return key
    if len(available_keys) == 1: return available_keys[0]
    raise KeyError(f"Could not determine IMU key. Available: {available_keys}")

def read_imu_data(path: Path) -> pd.DataFrame:
    key = str(find_imu_data_key(path))
    
    with h5py.File(path, 'r') as f:
        imu_object = f[key]
        
        if isinstance(imu_object, h5py.Group):
            if 'data' in imu_object and 'timestamps' in imu_object:
                data = imu_object['data'][()]
                timestamps = imu_object['timestamps'][()]
                full_data = np.column_stack([timestamps, data])
                return pd.DataFrame(full_data)
            else:
                best_name, best_shape = None, (0, 0)
                for name in imu_object.keys():
                    d = imu_object[name]
                    if isinstance(d, h5py.Dataset) and d.ndim == 2:
                        if d.shape[1] > best_shape[1]:
                            best_name, best_shape = name, d.shape
                if best_name:
                    return pd.DataFrame(imu_object[best_name][()])
                raise ValueError("No suitable dataset found")
        else:
            return pd.DataFrame(imu_object[()])

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
    
    Returns: Initial rotation matrix (3x3)
    """
    # Normalize accelerometer (should point opposite to gravity in body frame)
    down = normalize_vector(accel)
    
    # World frame: Z is up, so gravity points down (-Z)
    # Body frame: accelerometer at rest measures +Z (up)
    # So body Z-axis aligns with -down in world frame
    
    # Normalize magnetometer
    mag_norm = normalize_vector(mag)
    
    # Remove mag component along gravity direction
    mag_horizontal = mag_norm - np.dot(mag_norm, down) * down
    mag_horizontal = normalize_vector(mag_horizontal)
    
    # Build rotation matrix
    # Body Z = -down (up direction)
    # Body X = mag_horizontal (north direction)
    # Body Y = Z × X (east direction)
    
    z_body = -down
    x_body = mag_horizontal
    y_body = np.cross(z_body, x_body)
    y_body = normalize_vector(y_body)
    
    # Re-orthogonalize
    x_body = np.cross(y_body, z_body)
    x_body = normalize_vector(x_body)
    
    # Rotation matrix: columns are body axes in world frame
    R_init = np.column_stack([x_body, y_body, z_body])
    
    return R_init

def integrate_gyroscope_orientation(timestamps, gyro_data, accel_data, mag_data):
    """
    Track orientation over time using gyroscope integration with
    accelerometer/magnetometer corrections.
    
    This is a simplified complementary filter approach.
    
    Args:
        timestamps: (N,) array of timestamps in seconds
        gyro_data: (N, 3) array of gyroscope data in rad/s [gx, gy, gz]
        accel_data: (N, 3) array of accelerometer data in m/s²
        mag_data: (N, 3) array of magnetometer data in µT
    
    Returns:
        rotation_matrices: (N, 3, 3) array of rotation matrices
    """
    N = len(timestamps)
    rotation_matrices = np.zeros((N, 3, 3))
    
    # Initialize with first sample (assume at rest)
    rotation_matrices[0] = get_initial_orientation_from_accel_mag(
        accel_data[0], mag_data[0]
    )
    
    # Integration parameters
    alpha_accel = 0.02  # Accelerometer correction weight (2% per step)
    alpha_mag = 0.01  # Magnetometer correction (DISABLED - mag data unreliable)
    
    print(f"Integrating gyroscope data ({N} samples)...")
    
    for i in range(1, N):
        dt = timestamps[i] - timestamps[i-1]
        
        # Clamp dt to reasonable range
        if dt <= 0 or dt > 0.1:
            dt = 0.005  # Default 200 Hz
        
        # Current rotation
        R_current = rotation_matrices[i-1]
        
        # === STEP 1: Gyroscope Integration ===
        # Angular velocity in body frame
        omega_body = gyro_data[i]  # [gx, gy, gz] in rad/s
        
        # Rotation increment using Rodrigues formula (small angle approximation)
        angle = np.linalg.norm(omega_body) * dt
        
        if angle > 1e-6:
            axis = omega_body / np.linalg.norm(omega_body)
            # Rotation matrix from axis-angle
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            R_gyro = R_current @ dR
        else:
            R_gyro = R_current
        
        # === STEP 2: Accelerometer Correction ===
        # Expected gravity direction in world frame: [0, 0, -9.81]
        # Measured acceleration in body frame
        accel_body = accel_data[i]
        accel_mag = np.linalg.norm(accel_body)
        
        # Only use accelerometer correction if magnitude is close to gravity
        # (i.e., not during high acceleration)
        # if 8.0 < accel_mag < 11.0:  # Allow ±2 m/s² deviation
        #     # Expected down direction in body frame
        #     down_body_measured = normalize_vector(accel_body)
        #     down_body_expected = R_gyro.T @ np.array([0, 0, -1])
            
        #     # Correction axis and angle
        #     correction_axis = np.cross(down_body_expected, down_body_measured)
        #     correction_angle = np.arcsin(np.clip(np.linalg.norm(correction_axis), -1, 1))
            
        #     if correction_angle > 1e-6:
        #         correction_axis = normalize_vector(correction_axis)
        #         # Small correction
        #         correction_angle *= alpha_accel
                
        #         K = np.array([
        #             [0, -correction_axis[2], correction_axis[1]],
        #             [correction_axis[2], 0, -correction_axis[0]],
        #             [-correction_axis[1], correction_axis[0], 0]
        #         ])
        #         dR_accel = np.eye(3) + np.sin(correction_angle) * K + \
        #                   (1 - np.cos(correction_angle)) * (K @ K)
        #         R_gyro = R_gyro @ dR_accel
        
        # === STEP 3: Magnetometer Correction (DISABLED) ===
        # Magnetometer data is easily influenced by nearby metal/EM interference
        # Commenting out mag correction - relying on gyroscope for yaw
        
        # # Get horizontal plane magnetic field
        # mag_world = R_gyro @ mag_data[i]
        # mag_horizontal = np.array([mag_world[0], mag_world[1], 0])
        # mag_horizontal = normalize_vector(mag_horizontal)
        
        # # Expected north direction (from initial alignment)
        # R_init = rotation_matrices[0]
        # mag_init_world = R_init @ mag_data[0]
        # north_expected = normalize_vector(np.array([mag_init_world[0], mag_init_world[1], 0]))
        
        # # Yaw correction
        # yaw_error = np.arctan2(
        #     mag_horizontal[1] * north_expected[0] - mag_horizontal[0] * north_expected[1],
        #     mag_horizontal[0] * north_expected[0] + mag_horizontal[1] * north_expected[1]
        # )
        
        # # Apply small yaw correction
        # yaw_correction = yaw_error * alpha_mag
        # R_yaw_corr = np.array([
        #     [np.cos(yaw_correction), -np.sin(yaw_correction), 0],
        #     [np.sin(yaw_correction), np.cos(yaw_correction), 0],
        #     [0, 0, 1]
        # ])
        # R_gyro = R_yaw_corr @ R_gyro
        
        # Store result
        rotation_matrices[i] = R_gyro
        
        if i % 200 == 0:
            print(f"  Processed {i}/{N} samples...")
    
    return rotation_matrices

# --- RENDER SINGLE FRAME ---

def render_pose_frame(rotation_matrix, time_elapsed, frame_num, total_frames, arrow_length=0.5):
    """Render a single frame showing the sensor orientation"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fixed origin
    origin = np.array([0, 0, 0])
    
    # Extract axis directions from rotation matrix
    x_dir = rotation_matrix[:, 0] * arrow_length
    y_dir = rotation_matrix[:, 1] * arrow_length
    z_dir = rotation_matrix[:, 2] * arrow_length
    
    # Draw arrows
    ax.quiver(origin[0], origin[1], origin[2], 
             x_dir[0], x_dir[1], x_dir[2],
             color='red', arrow_length_ratio=0.15, linewidth=4, alpha=0.9, label='X-axis')
    ax.quiver(origin[0], origin[1], origin[2],
             y_dir[0], y_dir[1], y_dir[2],
             color='green', arrow_length_ratio=0.15, linewidth=4, alpha=0.9, label='Y-axis')
    ax.quiver(origin[0], origin[1], origin[2],
             z_dir[0], z_dir[1], z_dir[2],
             color='blue', arrow_length_ratio=0.15, linewidth=4, alpha=0.9, label='Z-axis')
    
    # Add origin point
    ax.scatter([0], [0], [0], c='black', s=100, marker='o', zorder=10)
    
    # Set fixed limits
    lim = arrow_length * 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(f'Sensor Orientation (Gyro-based)\nTime: {time_elapsed:.2f}s | Frame: {frame_num}/{total_frames}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

# --- CREATE GIF ---

def create_pose_gif(rotation_matrices, timestamps, output_path, arrow_length=0.5):
    """Create a GIF showing sensor orientation over time"""
    # Subsample frames
    if FRAME_SKIP > 1:
        indices = np.arange(0, len(rotation_matrices), FRAME_SKIP)
        rotation_matrices = rotation_matrices[indices]
        timestamps = timestamps[indices]
    
    num_frames = len(rotation_matrices)
    print(f"Creating GIF with {num_frames} frames...")
    
    # Render all frames
    frames = []
    for i, (R, t) in enumerate(zip(rotation_matrices, timestamps)):
        if i % 20 == 0:  # Progress indicator
            print(f"  Rendering frame {i+1}/{num_frames}...")
        
        time_elapsed = t - timestamps[0]
        img = render_pose_frame(R, time_elapsed, i+1, num_frames, arrow_length)
        frames.append(img)
    
    # Save as GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving GIF to: {output_path}")
    
    duration = int(1000 / FPS)  # milliseconds per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # 0 means infinite loop
    )
    
    print(f"✅ GIF saved successfully! ({num_frames} frames, {FPS} FPS)")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# --- MAIN ---

if __name__ == "__main__":
    target_path = Path(TARGET_FILE)
    
    if not target_path.exists():
        print(f"❌ File not found: {target_path}")
        sys.exit(1)
    
    print(f"🚀 Processing: {target_path.name}")
    print(f"Method: Gyroscope integration with accel/mag corrections")
    print()
    
    try:
        # Read data
        df = read_imu_data(target_path)
        df = df.dropna().reset_index(drop=True)
        
        if df.shape[1] < 10:
            raise ValueError(f"Expected at least 10 columns, got {df.shape[1]}")
        
        # Extract sensor data
        # Columns: [timestamp, ax, ay, az, gx, gy, gz, mx, my, mz]
        timestamps = df.iloc[:, 0].values
        accel_data = df.iloc[:, 1:4].values * 9.81  # Convert g to m/s²
        gyro_data = df.iloc[:, 4:7].values * (np.pi / 180)  # Convert deg/s to rad/s
        mag_data = df.iloc[:, 7:10].values  # µT
        
        print(f"📊 Total samples: {len(df)}")
        print(f"📊 Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"📊 Sample rate: ~{len(df) / (timestamps[-1] - timestamps[0]):.1f} Hz")
        print()
        
        # Calculate orientation using gyroscope integration
        rotation_matrices = integrate_gyroscope_orientation(
            timestamps, gyro_data, accel_data, mag_data
        )
        
        print(f"✅ Orientation calculated for all {len(rotation_matrices)} samples")
        print()
        
        # Create GIF
        output_path = Path(OUTPUT_DIR) / target_path.parent.name / (target_path.stem + "_pose_gyro.gif")
        create_pose_gif(rotation_matrices, timestamps, output_path, ARROW_LENGTH)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

