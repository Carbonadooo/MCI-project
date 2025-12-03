import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import h5py # Need this library to inspect the HDF5 file contents

# --- CONFIGURATION ---
# No need to set IMU_HDF5_KEY anymore, it will be found dynamically.
# ---------------------

# --- I. CORE HELPER FUNCTIONS ---

def normalize_vectors(x: pd.Series, y: pd.Series, z: pd.Series, eps: float = 1e-9):
    """Normalizes a set of 3D vectors (x, y, z components)."""
    V = np.vstack([x.values, y.values, z.values]).T.astype(float)
    norms = np.linalg.norm(V, axis=1)
    safe_norms = np.where(norms > eps, norms, np.nan)
    U = V / safe_norms[:, None]
    return U[:, 0], U[:, 1], U[:, 2], norms

# --- I. CORE HELPER FUNCTIONS (Modified Read Function) ---

def find_imu_data_key(path: Path) -> str:
    """
    (Function remains the same, assuming 'imu' is the correct high-level key)
    """
    with h5py.File(path, 'r') as f:
        available_keys = list(f.keys())
    
    if 'imu' in available_keys:
        return 'imu'
        
    for key in available_keys:
        if 'imu' in key.lower() or 'sensor' in key.lower() or 'data' in key.lower():
            return key
            
    if len(available_keys) == 1:
        return available_keys[0]

    raise KeyError(f"Could not automatically determine the IMU data key. Available keys are: {available_keys}.")


def read_imu_data(path: Path) -> pd.DataFrame:
    """
    Reads IMU data from an HDF5 file using an auto-detected key.
    Handles the new format where data and timestamps are stored separately in a Group.
    """
    
    # 1. Find the Key
    try:
        key = find_imu_data_key(path)
    except Exception as e:
        raise Exception(f"Key detection failed for {path.name}: {e}")
        
    key = str(key) 

    # 2. Read the Data with h5py (handles both old and new formats)
    with h5py.File(path, 'r') as f:
        imu_object = f[key]
        
        if isinstance(imu_object, h5py.Group):
            # NEW FORMAT: Check if it has 'data' and 'timestamps' keys
            if 'data' in imu_object and 'timestamps' in imu_object:
                data = imu_object['data'][()]
                timestamps = imu_object['timestamps'][()]
                
                # Data shape: (N, 9) with columns [ax, ay, az, gx, gy, gz, mx, my, mz]
                # Add timestamps as the first column
                full_data = np.column_stack([timestamps, data])
                imu_data = pd.DataFrame(full_data)
            else:
                # OLD FORMAT: Find best 2D dataset in the Group
                best_dataset_name = None
                best_shape = (0, 0)
                
                for name in imu_object.keys():
                    dataset = imu_object[name]
                    if isinstance(dataset, h5py.Dataset) and dataset.ndim == 2:
                        current_shape = dataset.shape
                        if current_shape[1] > best_shape[1]: 
                            best_shape = current_shape
                            best_dataset_name = name
                
                if best_dataset_name:
                    data_array = imu_object[best_dataset_name][()]
                    imu_data = pd.DataFrame(data_array)
                else:
                    raise ValueError(f"HDF5 key '{key}' is a Group but no suitable 2D dataset was found among its items: {list(imu_object.keys())}.")
                         
        elif isinstance(imu_object, h5py.Dataset):
            # Simple Dataset format
            data_array = imu_object[()]
            if data_array.ndim == 2:
                imu_data = pd.DataFrame(data_array)
            else:
                raise ValueError(f"Dataset has unexpected dimensions: {data_array.shape}")
        else:
            raise TypeError(f"Object at key '{key}' is neither a Dataset nor a Group, it is {type(imu_object)}.")

    # 3. Clean and Validate
    imu_data = imu_data.dropna()
    imu_data = imu_data.reset_index(drop=True)
    
    if imu_data.shape[1] < 10:
        raise ValueError(f"Expected at least 10 columns (timestamp + 9 sensor values), got {imu_data.shape[1]} after reading with key '{key}'.")
        
    print(f"   -> Key used: '{key}'. Read {len(imu_data)} samples with {imu_data.shape[1]} columns.")

    return imu_data

# --- II. RPY AND ROTATION MATRIX CALCULATION (Unchanged Logic) ---

def df_to_rpy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Roll, Pitch, and Yaw for all time steps. Returns (roll, pitch, yaw) in radians."""
    
    # Column format: [timestamp(0), ax(1), ay(2), az(3), gx(4), gy(5), gz(6), mx(7), my(8), mz(9)]
    # Accelerometer: columns 1-3
    # Magnetometer: columns 7-9
    Accx = df.iloc[:, 1]
    Accy = df.iloc[:, 2]
    Accz = df.iloc[:, 3]
    Magx = df.iloc[:, 7]
    Magy = df.iloc[:, 8]
    Magz = df.iloc[:, 9]
    
    acc_unit_x, acc_unit_y, acc_unit_z, _ = normalize_vectors(Accx, Accy, Accz)
    mag_unit_x, mag_unit_y, mag_unit_z, _ = normalize_vectors(Magx, Magy, Magz)

    pitch = np.arcsin(-acc_unit_x) 
    roll = np.arctan2(acc_unit_y, acc_unit_z)

    sx, sy, sz = mag_unit_x, mag_unit_y, mag_unit_z
    theta, phi = pitch, roll

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    y_comp = sy * cos_p + sz * sin_p
    x_comp = sx * cos_t + sy * sin_p * sin_t - sz * cos_p * sin_t

    yaw = np.arctan2(y_comp, x_comp)
    
    return roll, pitch, yaw


def rpy_to_Rot(r: np.ndarray, p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Converts RPY arrays to an (N, 3, 3) NumPy array of Rotation Matrices (Z-Y-X convention)."""
    
    cos_y, sin_y = np.cos(y), np.sin(y)
    cos_p, sin_p = np.cos(p), np.sin(p)
    cos_r, sin_r = np.cos(r), np.sin(r)
    
    N = len(r)
    Rot = np.empty((N, 3, 3), dtype=np.float64)

    # R = R_yaw * R_pitch * R_roll
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

# --- III. MAIN BATCH PROCESSING LOGIC ---

def process_single_file(hdf5_path: Path, output_dir: Path):
    """Handles the processing and saving for a single HDF5 file."""
    
    print(f"\n--- Processing: {hdf5_path.name} ---")
    
    # 1. Feature Extraction
    try:
        # read_imu_data now handles key detection
        df = read_imu_data(hdf5_path) 
        roll, pitch, yaw = df_to_rpy(df)
        
        # Filter out NaN values in RPY
        valid_mask = np.isfinite(roll) & np.isfinite(pitch) & np.isfinite(yaw)
        if (~valid_mask).sum() > 0:
            print(f"   -> Filtering {(~valid_mask).sum()} invalid RPY samples out of {len(roll)}")
        
        roll = roll[valid_mask]
        pitch = pitch[valid_mask]
        yaw = yaw[valid_mask]
        
        rotation_matrices = rpy_to_Rot(roll, pitch, yaw)
    except Exception as e:
        print(f"❌ FAILED to process {hdf5_path.name}: {e}")
        return

    # 2. Save the Output
    output_filename = hdf5_path.stem + '_pose.npy'
    
    # Determine the task sub-folder based on the file's parent directory
    try:
        task_name = hdf5_path.parent.name
    except IndexError:
        task_name = ''

    final_output_dir = output_dir / task_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = final_output_dir / output_filename
    
    np.save(output_path, rotation_matrices)
    
    print(f"✅ Extracted {rotation_matrices.shape[0]} poses.")
    print(f"   Saved to: {output_path}")


def batch_process_data(root_data_dir: str, root_output_dir: str):
    """
    Finds all HDF5 files recursively and processes them.
    """
    root_data_path = Path(root_data_dir)
    root_output_path = Path(root_output_dir)
    
    if not root_data_path.is_dir():
        print(f"❌ Error: Data directory not found at {root_data_dir}")
        return

    hdf5_files = list(root_data_path.glob('**/*.hdf5'))
    
    if not hdf5_files:
        print(f"⚠️ No HDF5 files found in {root_data_dir} or its subdirectories.")
        return

    print("="*60)
    print(f"Starting batch processing of {len(hdf5_files)} files...")
    print(f"Output features will be saved to: {root_output_dir}")
    print("="*60)
    
    for hdf5_file in hdf5_files:
        process_single_file(hdf5_file, root_output_path)
        
    print("\n" + "="*60)
    print("Batch processing completed!")
    print("="*60)


if __name__ == "__main__":
    
    # You may need to install h5py: pip install h5py
    
    # 1. 📂 Define Input and Output Directories
    INPUT_ROOT = '/home/chuye/Documents/MCI-project/data/nov2_set' 
    OUTPUT_ROOT = '/home/chuye/Documents/MCI-project/data/nov2_set_pose_features' 

    # 2. 🚀 Run the Batch Processing
    batch_process_data(INPUT_ROOT, OUTPUT_ROOT)