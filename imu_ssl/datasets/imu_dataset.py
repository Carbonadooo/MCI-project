import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class IMUDataset(Dataset):
    """
    Minimal IMU-only dataset for SSL training.
    Loads IMU (T, 9) and pads/trims to target_seq_len.
    """
    def __init__(
        self,
        file_list,
        target_seq_len=1440,
        imu_normalizer=None,
    ):
        self.files = file_list
        self.target_seq_len = target_seq_len
        self.normalizer = imu_normalizer

    def __len__(self):
        return len(self.files)

    def _load_hdf5(self, path):
        """Load only IMU data from hdf5."""
        with h5py.File(path, "r") as f:
            imu = f["imu"]["data"][:]      # (T, 9)
            ts  = f["imu"]["timestamps"][:]  # still needed for trim-length consistency
        return imu, ts

    def _apply_normalization(self, imu):
        if self.normalizer is None:
            return imu
        mean = self.normalizer["mean"]
        std = self.normalizer["std"]
        return (imu - mean) / (std + 1e-8)

    def _pad_or_trim(self, imu, ts):
        if self.target_seq_len is None:
            return imu

        T = imu.shape[0]
        target = self.target_seq_len

        # Trim (center crop)
        if T > target:
            start = (T - target) // 2
            return imu[start:start+target]

        # Pad zeros
        if T < target:
            pad_len = target - T
            imu_pad = np.pad(imu, ((0, pad_len), (0, 0)), mode="constant")
            return imu_pad

        return imu

    def __getitem__(self, idx):
        path = self.files[idx]

        imu, ts = self._load_hdf5(path)
        imu = self._apply_normalization(imu)
        imu = self._pad_or_trim(imu, ts)

        return torch.from_numpy(imu).float()   # (T, 9)


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
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
