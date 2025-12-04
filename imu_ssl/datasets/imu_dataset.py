import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class IMUDataset(Dataset):
    """
    A dataset for loading IMU-only data from HDF5 files.
    
    Returns:
        imu: (T, 9) tensor
        imu_timestamps: (T,) tensor
        meta: dictionary with file path, frequencies, etc.
    """
    def __init__(
        self,
        file_list,
        target_seq_len=1440,
        imu_normalizer=None,
        load_video_meta=True,
    ):
        """
        Args:
            file_list (list[str]): Paths to HDF5 files.
            target_seq_len (int or None): If provided, sequences are padded or trimmed.
            imu_normalizer: dict with {"mean": np.array(9), "std": np.array(9)}.
            load_video_meta: Whether to load video timestamps (for visualization).
        """
        self.files = file_list
        self.target_seq_len = target_seq_len
        self.normalizer = imu_normalizer
        self.load_video_meta = load_video_meta

    def __len__(self):
        return len(self.files)

    def _load_hdf5(self, path):
        """Load IMU (and optionally video) from hdf5."""
        with h5py.File(path, "r") as f:
            # Load IMU data
            imu_data = f["imu"]["data"][:]      # shape (N, 9)
            imu_ts = f["imu"]["timestamps"][:]  # shape (N,)
            imu_freq = f["imu"].attrs["frequency"]

            # Optional: video metadata used for visualization
            if self.load_video_meta:
                video_ts = f["video"]["timestamps"][:]
                video_meta = dict(
                    fps=f["video"].attrs["fps"],
                    resolution=f["video"].attrs["resolution"],
                )
            else:
                video_ts = None
                video_meta = None

        return imu_data, imu_ts, imu_freq, video_ts, video_meta

    def _apply_normalization(self, imu):
        """Normalize IMU data (accelerometer, gyro, magnetometer)."""
        if self.normalizer is None:
            return imu
        mean = self.normalizer["mean"]
        std = self.normalizer["std"]
        return (imu - mean) / (std + 1e-8)

    def _pad_or_trim(self, imu, ts):
        """
        Pad or trim the sequence to target_seq_len.
        """
        if self.target_seq_len is None:
            return imu, ts

        T = imu.shape[0]
        target = self.target_seq_len

        # Too long → center crop
        if T > target:
            start = (T - target) // 2
            end = start + target
            return imu[start:end], ts[start:end]

        # Too short → pad zeros
        if T < target:
            pad_len = target - T
            imu_pad = np.pad(imu, ((0, pad_len), (0, 0)), mode="constant")
            ts_pad = np.pad(ts, (0, pad_len), mode="edge")
            return imu_pad, ts_pad

        # Just right
        return imu, ts

    def __getitem__(self, idx):
        path = self.files[idx]

        imu, imu_ts, imu_freq, video_ts, video_meta = self._load_hdf5(path)

        # Normalize
        imu = self._apply_normalization(imu)

        # Pad / Trim
        imu, imu_ts = self._pad_or_trim(imu, imu_ts)

        # Convert to torch
        imu = torch.from_numpy(imu).float()
        imu_ts = torch.from_numpy(imu_ts).float()

        sample = {
            "imu": imu,                  # (T, 9)
            "imu_timestamps": imu_ts,    # (T,)
            "imu_freq": imu_freq,
            "file": path,
        }

        # Optional video metadata for visualization
        if self.load_video_meta:
            sample.update({
                "video_timestamps": video_ts,
                "video_meta": video_meta,
            })

        return sample

def build_dataloader(
    file_list,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    target_seq_len=None,
    imu_normalizer=None,
):
    dataset = IMUDataset(
        file_list=file_list,
        target_seq_len=target_seq_len,
        imu_normalizer=imu_normalizer,
        load_video_meta=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
