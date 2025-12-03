import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

# --- CONFIGURATION ---
TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/punch_forward/punch_forward_20251108_204358.hdf5'
OUTPUT_DIR = '/home/chuye/Documents/MCI-project/data/pose_animations'
FPS = 30  # Frames per second for the animation
ARROW_LENGTH = 0.5  # Length of orientation arrows
FRAME_SKIP = 5  # Show every Nth frame (to speed up animation)
# ---------------------

# --- HELPER FUNCTIONS ---

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

def df_to_rpy(df: pd.DataFrame) -> tuple:
    df = df.dropna().reset_index(drop=True)
    if df.shape[1] < 10:
        raise ValueError(f"Expected at least 10 cols (with timestamp), got {df.shape[1]}")
    
    acc_unit_x, acc_unit_y, acc_unit_z, _ = normalize_vectors(
        df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3])
    mag_unit_x, mag_unit_y, mag_unit_z, _ = normalize_vectors(
        df.iloc[:, 7], df.iloc[:, 8], df.iloc[:, 9])
    
    pitch = np.arcsin(-acc_unit_x)
    roll = np.arctan2(acc_unit_y, acc_unit_z)
    
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

# --- ANIMATION FUNCTION ---

def create_pose_animation(rotation_matrices, timestamps, output_path, arrow_length=0.5):
    """
    Create an MP4 animation showing sensor orientation over time.
    Origin is fixed at (0, 0, 0).
    """
    # Subsample frames if needed
    if FRAME_SKIP > 1:
        indices = np.arange(0, len(rotation_matrices), FRAME_SKIP)
        rotation_matrices = rotation_matrices[indices]
        timestamps = timestamps[indices]
    
    print(f"Creating animation with {len(rotation_matrices)} frames...")
    
    # Setup figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fixed origin
    origin = np.array([0, 0, 0])
    
    # Initialize plot elements
    x_arrow = ax.quiver(0, 0, 0, 1, 0, 0, color='red', 
                        arrow_length_ratio=0.15, linewidth=3, label='X-axis')
    y_arrow = ax.quiver(0, 0, 0, 0, 1, 0, color='green',
                        arrow_length_ratio=0.15, linewidth=3, label='Y-axis')
    z_arrow = ax.quiver(0, 0, 0, 0, 0, 1, color='blue',
                        arrow_length_ratio=0.15, linewidth=3, label='Z-axis')
    
    # Time text
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, 
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set fixed axis limits
    lim = arrow_length * 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Sensor Orientation Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    
    def update(frame):
        """Update function for animation"""
        # Get rotation matrix for this frame
        R = rotation_matrices[frame]
        
        # Extract axis directions (columns of rotation matrix)
        x_dir = R[:, 0] * arrow_length
        y_dir = R[:, 1] * arrow_length
        z_dir = R[:, 2] * arrow_length
        
        # Remove old arrows
        ax.collections.clear()
        
        # Draw new arrows from origin
        ax.quiver(origin[0], origin[1], origin[2], 
                 x_dir[0], x_dir[1], x_dir[2],
                 color='red', arrow_length_ratio=0.15, linewidth=3, alpha=0.9)
        ax.quiver(origin[0], origin[1], origin[2],
                 y_dir[0], y_dir[1], y_dir[2],
                 color='green', arrow_length_ratio=0.15, linewidth=3, alpha=0.9)
        ax.quiver(origin[0], origin[1], origin[2],
                 z_dir[0], z_dir[1], z_dir[2],
                 color='blue', arrow_length_ratio=0.15, linewidth=3, alpha=0.9)
        
        # Update time text
        time_elapsed = timestamps[frame] - timestamps[0]
        time_text.set_text(f'Time: {time_elapsed:.2f}s\nFrame: {frame}/{len(rotation_matrices)}')
        
        return ax.collections + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(rotation_matrices),
                        interval=1000/FPS, blit=False, repeat=True)
    
    # Save as GIF (fallback if ffmpeg not available)
    # Change extension to .gif
    if output_path.suffix == '.mp4':
        output_path = output_path.with_suffix('.gif')
    
    writer = PillowWriter(fps=FPS, metadata={'artist': 'IMU Pose Visualization'})
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving animation to: {output_path}")
    print(f"   (Saving as GIF since ffmpeg is not installed)")
    anim.save(str(output_path), writer=writer)
    print(f"✅ Animation saved successfully!")
    
    plt.close(fig)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    target_path = Path(TARGET_FILE)
    
    if not target_path.exists():
        print(f"❌ File not found: {target_path}")
        sys.exit(1)
    
    print(f"🚀 Processing: {target_path.name}")
    
    try:
        # 1. Read and process data
        df = read_imu_data(target_path)
        r, p, y = df_to_rpy(df)
        
        # Clean NaNs
        valid_mask = np.isfinite(r) & np.isfinite(p) & np.isfinite(y)
        if (~valid_mask).sum() > 0:
            print(f"⚠️  Filtering {(~valid_mask).sum()} invalid RPY samples")
        
        r, p, y = r[valid_mask], p[valid_mask], y[valid_mask]
        df = df[valid_mask].reset_index(drop=True)
        
        # Get rotation matrices
        rot = rpy_to_Rot(r, p, y)
        timestamps = df.iloc[:, 0].values
        
        print(f"📊 Total samples: {len(rot)}")
        print(f"📊 Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"📊 Animation frames: {len(rot) // FRAME_SKIP}")
        
        # 2. Create animation
        output_path = Path(OUTPUT_DIR) / target_path.parent.name / (target_path.stem + "_pose.mp4")
        create_pose_animation(rot, timestamps, output_path, ARROW_LENGTH)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

