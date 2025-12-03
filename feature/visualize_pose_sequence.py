import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys
from PIL import Image
import io

# --- CONFIGURATION ---
# TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/punch_forward/punch_forward_20251108_204358.hdf5'
TARGET_FILE = '/home/chuye/Documents/MCI-project/data/nov2_set/right/right_20251202_205538.hdf5'
OUTPUT_DIR = '/home/chuye/Documents/MCI-project/data/pose_animations'
FPS = 30  # Frames per second for the GIF
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

def df_to_rpy(df: pd.DataFrame) -> tuple:
    df = df.dropna().reset_index(drop=True)
    if df.shape[1] < 10:
        raise ValueError(f"Expected at least 10 cols, got {df.shape[1]}")
    
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
    ax.set_title(f'Sensor Orientation\nTime: {time_elapsed:.2f}s | Frame: {frame_num}/{total_frames}',
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
    
    try:
        # Read and process data
        df = read_imu_data(target_path)
        r, p, y = df_to_rpy(df)
        
        # Clean NaNs
        valid_mask = np.isfinite(r) & np.isfinite(p) & np.isfinite(y)
        if (~valid_mask).sum() > 0:
            print(f"⚠️  Filtering {(~valid_mask).sum()} invalid samples")
        
        r, p, y = r[valid_mask], p[valid_mask], y[valid_mask]
        df = df[valid_mask].reset_index(drop=True)
        
        # Get rotation matrices
        rot = rpy_to_Rot(r, p, y)
        timestamps = df.iloc[:, 0].values
        
        print(f"📊 Total samples: {len(rot)}")
        print(f"📊 Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"📊 Output frames: {len(rot) // FRAME_SKIP}")
        
        # Create GIF
        output_path = Path(OUTPUT_DIR) / target_path.parent.name / (target_path.stem + "_pose.gif")
        create_pose_gif(rot, timestamps, output_path, ARROW_LENGTH)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

