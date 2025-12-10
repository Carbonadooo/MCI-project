import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from imu_ssl.datasets.imu_preprocess_utils import compute_fe_features
from imu_ssl.datasets.imu_augmentation import (
    Compose, Jitter, Scaling, RandomCrop, TimeWarp, ChannelDropout, TimeStretch, Shift
)


class IMUDataset(Dataset):
    """
    SimCLR-style IMU dataset for contrastive self-supervised learning.
    Loads IMU (T, 9), pads/trims to target_seq_len, and returns two augmented views.
    """
    def __init__(
        self,
        file_list,
        target_seq_len=1000,
        imu_normalizer=None,
        augment=None,
        expand_factor=100,
    ):
        self.files = file_list
        self.target_seq_len = target_seq_len
        self.normalizer = imu_normalizer
        self.augment = augment
        self.expand_factor = expand_factor  

    def __len__(self):
        return len(self.files) * self.expand_factor

    def _load_hdf5(self, path):
        """Load only IMU data from hdf5."""
        with h5py.File(path, "r") as f:
            imu = f["imu"]["data"][:]      # (T, 9)
            ts  = f["imu"]["timestamps"][:]  # still needed for trim-length consistency

        # ---------- Check for NaN ----------
        if np.isnan(imu).any():
            nan_mask = np.isnan(imu)

            # Logging: report how many NaNs & file path
            num_nan = nan_mask.sum()
            # print(f"[WARN] NaN detected in file: {path}  (count={num_nan})")

            # ---------- Fix NaNs by interpolation per channel ----------
            T, C = imu.shape
            for c in range(C):
                channel = imu[:, c]
                idx = np.arange(T)

                nan_idx = np.isnan(channel)
                if nan_idx.any():
                    valid_idx = idx[~nan_idx]
                    valid_vals = channel[~nan_idx]

                    # If entire channel is NaN, fill with zeros
                    if len(valid_idx) == 0:
                        print(f"[WARN] Channel {c} in {path} is fully NaN. Filling with zeros.")
                        imu[:, c] = 0.0
                        continue

                    # Otherwise do 1D interpolation
                    imu[:, c] = np.interp(idx, valid_idx, valid_vals)

        return imu, ts

    def _apply_normalization(self, imu):
        if self.normalizer is None:
            return imu
        mean = self.normalizer["mean"]
        std = self.normalizer["std"]
        return (imu - mean) / (std + 1e-8)
    
    def _resize_to_len(self, imu, target_len):
        T, C = imu.shape
        if T == target_len:
            return imu

        idx_old = np.linspace(0.0, 1.0, T)
        idx_new = np.linspace(0.0, 1.0, target_len)
        out = np.zeros((target_len, C), dtype=imu.dtype)
        for c in range(C):
            out[:, c] = np.interp(idx_new, idx_old, imu[:, c])
        return out

    def trim_to_target_length(self, imu, ts, target):
        jerk_thresh = 0.05
        T = imu.shape[0]
        if T == target:
            return imu
        elif T > target:
            jerk = imu[:, 3:6]
            jerk_norm = np.linalg.norm(jerk, axis=1)
            left = 0
            right = T

            while (right - left) > target:
                if jerk_norm[right - 1] < jerk_thresh:
                    right -= 1
                else:
                    break
            while (right - left) > target:
                if jerk_norm[left] < jerk_thresh:
                    left += 1
                else:
                    break

            cropped = imu[left:right]
            L = cropped.shape[0]
            if L == target:
                return cropped
            if L > target:
                return self._resize_to_len(cropped, target)
        elif T < target:
            return self._resize_to_len(imu, target  )


    def dataset_augment(self):
        return Compose([
            Scaling(0.1),
            TimeWarp(0.2),
            TimeStretch((0.7, 1.3)),
            ChannelDropout(0.05),
        ])

    def remove_spikes(self, imu):
        """
        Remove spikes in accelerometer channels (0-2) using median filter.
        """
        from scipy.signal import medfilt
        
        acc_channels = [0, 1, 2]
        
        for ch in acc_channels:
            # Apply median filter of size 4 to remove spikes
            imu[:, ch] = medfilt(imu[:, ch], kernel_size=5)  # Use size 5 (closest odd number to 4)
        
        return imu

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        low  = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def __getitem__(self, idx):
        real_idx = idx % len(self.files)
        path = self.files[real_idx]
        imu, ts = self._load_hdf5(path) # Raw data
        imu = self.remove_spikes(imu) # Remove abrupt spikes in acc channels
        imu[:, 0:6] = self.butter_bandpass_filter(data=imu[:, 0:6], lowcut=0.3, highcut=8, fs=200) # Band-pass filter
        # imu = compute_fe_features(imu) # Add features
        imu = self.trim_to_target_length(imu, ts, self.target_seq_len) # Align length
        imu = self._apply_normalization(imu) # Normalize
        dataset_augment = self.dataset_augment() # Augment dataset
        imu = dataset_augment(imu.copy())

        # --- SimCLR augmentation: return two augmented views ---
        if self.augment is not None:
            x1 = self.augment(imu)
            x2 = self.augment(imu)
            return torch.from_numpy(x1).float(), torch.from_numpy(x2).float()
        else:
            tensor = torch.from_numpy(imu).float()
            return tensor, tensor


def build_dataloader(
    file_list,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    target_seq_len=None,
    imu_normalizer=None,
    augment=None,
    expand_factor=20,
):
    """
    Build SimCLR-style dataloader that returns pairs of augmented views.
    Returns: DataLoader that yields (x1, x2) tuples where both are (B, T, 9)
    """
    dataset = IMUDataset(
        file_list=file_list,
        target_seq_len=target_seq_len,
        imu_normalizer=imu_normalizer,
        augment=augment,
        expand_factor=expand_factor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
