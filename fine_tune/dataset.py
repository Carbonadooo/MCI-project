"""
IMU-Text Dataset for fine-tuning

Handles loading HDF5 files, IMU data augmentation, and text augmentation
"""
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from inference_hdf5 import load_hdf5_imu
from fine_tune.text_alternatives import TEXT_ALTERNATIVES
from lib.rotation_utils import compute_rotation_features, compute_rotation_features_fast, compute_world_frame_acceleration


def augment_imu_data(imu_data, training=True, config=None):
    """
    Apply data augmentation to IMU data
    
    Args:
        imu_data: numpy array (6, N) - should be clean (no NaN/Inf)
        training: if True, apply augmentation; if False, return as-is
        config: dict with augmentation parameters (optional)
        
    Returns:
        augmented IMU data (6, N)
    """
    if not training:
        return imu_data
    
    # Default parameters
    if config is None:
        config = {
            'scale_min': 0.7,
            'scale_max': 1.3,
            'noise_min': 0.001,
            'noise_max': 0.003,
            'zero_pad_max': 0.10
        }
    
    imu_aug = imu_data.copy()
    
    # 1. Random scaling
    scale = random.uniform(config['scale_min'], config['scale_max'])
    imu_aug = imu_aug * scale
    
    # 2. Add small Gaussian noise (relative to signal magnitude)
    signal_std = np.std(imu_aug, axis=1, keepdims=True)
    signal_std = np.maximum(signal_std, 1e-8)  # Avoid division by zero
    noise_factor = random.uniform(config['noise_min'], config['noise_max'])
    noise = np.random.randn(*imu_aug.shape) * signal_std * noise_factor
    imu_aug = imu_aug + noise
    
    # # 3. Random start zero-padding
    # n_samples = imu_aug.shape[1]
    # start_zero_pct = random.uniform(0, config['zero_pad_max'])
    # start_zero_len = int(n_samples * start_zero_pct)
    # if start_zero_len > 0:
    #     imu_aug[:, :start_zero_len] = 0.0
    
    # # 4. Random end zero-padding
    # end_zero_pct = random.uniform(0, config['zero_pad_max'])
    # end_zero_len = int(n_samples * end_zero_pct)
    # if end_zero_len > 0:
    #     imu_aug[:, -end_zero_len:] = 0.0
    
    return imu_aug


class IMUTextDataset(Dataset):
    """
    Dataset for IMU-Text pairs from folder structure
    
    Loads HDF5 files from a directory with class-named folders.
    Applies IMU and text augmentation during training.
    """
    
    def __init__(self, data_dir, window_size=1000, augmentation_multiplier=50, 
                 training=True, max_per_class=None, aug_config=None, num_channels=6):
        """
        Args:
            data_dir: Root directory with class folders containing HDF5 files
            window_size: Fixed window size in samples (1000 = 5 seconds at 200 Hz)
            augmentation_multiplier: Number of augmented versions per original sample
            training: If True, apply augmentation; if False, use original data only
            max_per_class: Limit samples per class (for testing)
            aug_config: Dict with augmentation parameters (optional)
            num_channels: Number of IMU channels to load (6 or 9)
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.augmentation_multiplier = augmentation_multiplier if training else 1
        self.training = training
        self.aug_config = aug_config
        self.num_channels = num_channels
        self.samples = []
        self._logged_nan_files = set()  # Track which files we've logged NaN warnings for
        
        # Scan directory structure
        class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        
        print(f"Loading dataset from {data_dir}")
        print(f"Found {len(class_dirs)} classes")
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            hdf5_files = sorted(glob.glob(str(class_dir / "*.hdf5")))
            
            if max_per_class:
                hdf5_files = hdf5_files[:max_per_class]
            
            print(f"  {class_name}: {len(hdf5_files)} samples")
            
            for hdf5_file in hdf5_files:
                self.samples.append({
                    'file': hdf5_file,
                    'class_name': class_name,
                    'text': class_name.replace('_', ' ')  # "clap_once" -> "clap once"
                })
        
        print(f"Original samples: {len(self.samples)}")
        
        # Expand dataset with augmentation multiplier
        # Training: Each original sample will have N augmented versions
        # Validation: No expansion (augmentation_multiplier=1)
        original_samples = self.samples.copy()
        self.samples = []
        for aug_idx in range(self.augmentation_multiplier):
            for sample in original_samples:
                augmented_sample = sample.copy()
                augmented_sample['aug_idx'] = aug_idx  # Track which augmentation this is
                self.samples.append(augmented_sample)
        
        if self.training:
            print(f"After {self.augmentation_multiplier}x augmentation: {len(self.samples)} total samples")
        else:
            print(f"Validation mode (no augmentation): {len(self.samples)} total samples")
        
        # Get unique class names
        self.class_names = sorted(list(set([s['class_name'] for s in original_samples])))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Use first alternative as canonical text label
        self.text_labels = []
        for name in self.class_names:
            if name in TEXT_ALTERNATIVES:
                self.text_labels.append(TEXT_ALTERNATIVES[name][0])  # First is canonical
            else:
                self.text_labels.append(name.replace('_', ' '))
        
        print(f"Unique classes: {len(self.class_names)}")
        print(f"Text augmentation: {sum(1 for name in self.class_names if name in TEXT_ALTERNATIVES)} classes with alternatives")
    
    def __len__(self):
        return len(self.samples)
    
    def _interpolate_nans(self, data):
        """
        Interpolate NaN values in IMU data channel-wise
        
        Args:
            data: numpy array (6, N) with potential NaN values
            
        Returns:
            data with NaNs interpolated
        """
        # Interpolate channel by channel
        for channel_idx in range(data.shape[0]):
            channel_data = data[channel_idx, :]
            
            if np.isnan(channel_data).any():
                # Find NaN positions
                nan_mask = np.isnan(channel_data)
                
                # If all values are NaN, fill with zeros
                if nan_mask.all():
                    channel_data[:] = 0.0
                    continue
                
                # Get valid (non-NaN) indices and values
                valid_indices = np.where(~nan_mask)[0]
                valid_values = channel_data[~nan_mask]
                
                # Interpolate NaN values
                if len(valid_indices) > 1:
                    # Use linear interpolation for interior NaNs
                    channel_data[nan_mask] = np.interp(
                        np.where(nan_mask)[0],
                        valid_indices,
                        valid_values
                    )
                else:
                    # If only one valid value, fill NaNs with that value
                    channel_data[nan_mask] = valid_values[0] if len(valid_values) > 0 else 0.0
                
                data[channel_idx, :] = channel_data
        
        return data
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load IMU data
        try:
            data = load_hdf5_imu(sample['file'], verbose=False, resample_to_200hz=True, num_channels=self.num_channels)
            imu_data = data['imu_data']  # Shape: (num_channels, N)
            
            # Check for NaN and interpolate or skip
            if np.isnan(imu_data).any():
                nan_count = np.isnan(imu_data).sum()
                nan_threshold = int(0.15 * imu_data.size)
                file_name = sample['file']
                
                # Only log once per file (and only for validation dataset to reduce spam)
                if file_name not in self._logged_nan_files:
                    self._logged_nan_files.add(file_name)
                    
                    if nan_count > nan_threshold:
                        # Too many NaNs - skip this sample
                        print(f"\n⚠️  SKIP: {Path(file_name).name} - {nan_count}/{imu_data.size} NaNs ({100*nan_count/imu_data.size:.1f}%)")
                    elif not self.training:  # Only log interpolation for validation (less spam)
                        # Few NaNs - interpolate them
                        print(f"✓ Interpolated: {Path(file_name).name} - {nan_count} NaNs ({100*nan_count/imu_data.size:.2f}%)")
                
                if nan_count > nan_threshold:
                    # Too many NaNs - return dummy sample
                    return {
                        'imu': torch.zeros(self.num_channels, self.window_size),
                        'class_idx': self.class_to_idx[sample['class_name']],
                        'text': sample['text']
                    }
                else:
                    # Few NaNs - interpolate them
                    imu_data = self._interpolate_nans(imu_data)
            
            # Check for Inf
            if np.isinf(imu_data).any():
                print(f"\n❌ ERROR: Inf in loaded HDF5 file: {sample['file']}")
                imu_data = np.nan_to_num(imu_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract random window of fixed size
            n_samples = imu_data.shape[1]
            
            if n_samples >= self.window_size:
                # Random start position (training) or center (validation)
                if self.training:
                    start_idx = np.random.randint(0, n_samples - self.window_size + 1)
                else:
                    # Use center window for validation (more consistent)
                    start_idx = (n_samples - self.window_size) // 2
                window = imu_data[:, start_idx:start_idx + self.window_size]
            else:
                # Pad if too short
                padding = self.window_size - n_samples
                window = np.pad(imu_data, ((0, 0), (0, padding)), mode='constant')
            
            # Apply IMU data augmentation (only if training)
            window = augment_imu_data(window, training=self.training, config=self.aug_config)
            
            # Compute rotation features if using 18-channel or 21-channel mode
            if self.num_channels == 18 or self.num_channels == 21:
                # Window should be (9, N) at this point
                if window.shape[0] != 9:
                    raise ValueError(f"{self.num_channels}-channel mode requires 9-channel input, got {window.shape[0]}")
                
                try:
                    # Compute rotation matrix features (9, N)
                    # rotation_features = compute_rotation_features_fast(window, sample_rate=200)
                    rotation_features = compute_rotation_features(window, sample_rate=200)
                    
                    if self.num_channels == 18:
                        # Concatenate: (9, N) IMU + (9, N) rotation -> (18, N)
                        window = np.concatenate([window, rotation_features], axis=0)
                    
                    elif self.num_channels == 21:
                        # Also compute world-frame linear acceleration
                        world_accel = compute_world_frame_acceleration(
                            window, rotation_features, sample_rate=200
                        )
                        # Concatenate: (9, N) IMU + (9, N) rotation + (3, N) world accel -> (21, N)
                        window = np.concatenate([window, rotation_features, world_accel], axis=0)
                        
                except Exception as e:
                    # Fallback: use zeros for rotation features if computation fails
                    print(f"⚠️  Rotation feature computation failed: {e}, using zeros")
                    rotation_features = np.zeros((9, window.shape[1]))
                    if self.num_channels == 21:
                        world_accel = np.zeros((3, window.shape[1]))
                        window = np.concatenate([window, rotation_features, world_accel], axis=0)
                    else:
                        window = np.concatenate([window, rotation_features], axis=0)
            
            # Convert to tensor
            imu_tensor = torch.from_numpy(window).float()
            
            # Get class index
            class_idx = self.class_to_idx[sample['class_name']]
            
            # Get text with augmentation (only during training)
            class_name = sample['class_name']
            if self.training and class_name in TEXT_ALTERNATIVES:
                # Training: randomly pick one alternative
                text = random.choice(TEXT_ALTERNATIVES[class_name])
            elif class_name in TEXT_ALTERNATIVES:
                # Validation: use canonical (first) alternative
                text = TEXT_ALTERNATIVES[class_name][0]
            else:
                # Fallback to default (replace underscores with spaces)
                text = sample['text']
            
            return {
                'imu': imu_tensor,
                'class_idx': class_idx,
                'text': text
            }
            
        except Exception as e:
            print(f"Error loading {sample['file']}: {e}")
            # Return a dummy sample
            return {
                'imu': torch.zeros(self.num_channels, self.window_size),
                'class_idx': 0,
                'text': 'error'
            }

