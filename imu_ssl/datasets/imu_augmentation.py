import numpy as np
import torch
import random
from scipy.interpolate import CubicSpline


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Jitter:
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(0, self.sigma, x.shape)
        return x + noise

class Scaling:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(1.0, self.sigma)
        return x * factor

class RandomCrop:
    def __init__(self, crop_ratio=0.9):
        self.crop_ratio = crop_ratio

    def __call__(self, x):
        T = len(x)
        new_T = int(T * self.crop_ratio)

        if new_T >= T:
            return x

        start = np.random.randint(0, T - new_T)
        cropped = x[start:start+new_T]

        # resize back to original length by interpolation
        idx_old = np.linspace(0, 1, new_T)
        idx_new = np.linspace(0, 1, T)
        out = np.zeros_like(x)
        for c in range(x.shape[1]):
            out[:, c] = np.interp(idx_new, idx_old, cropped[:, c])

        return out

class TimeWarp:
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, x):
        T = len(x)
        tt = np.arange(T)

        # random smooth curve
        random_curve = np.random.normal(loc=1.0, scale=self.sigma, size=2)
        control_points = np.array([random_curve[0], random_curve[1]])
        spline = CubicSpline([0, T-1], control_points)
        warp = spline(tt)

        tt_new = tt * warp
        tt_new = np.clip(tt_new, 0, T - 1)

        out = np.zeros_like(x)
        for c in range(x.shape[1]):
            out[:, c] = np.interp(tt, tt_new, x[:, c])

        return out

# class Permutation:
#     def __init__(self, num_segs=4):
#         self.num_segs = num_segs

#     def __call__(self, x):
#         T = len(x)
#         seg = T // self.num_segs
#         segments = [x[i*seg:(i+1)*seg] for i in range(self.num_segs)]
#         random.shuffle(segments)
#         return np.concatenate(segments, axis=0)

class ChannelDropout:
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, x):
        x = x.copy()
        for c in range(x.shape[1]):
            if random.random() < self.drop_prob:
                x[:, c] = 0
        return x

class TimeStretch:
    def __init__(self, rate_range=(0.8, 1.2)):
        self.rate_range = rate_range

    def __call__(self, x):
        rate = np.random.uniform(*self.rate_range)
        T = x.shape[0]

        old_idx = np.arange(T)
        new_T = int(T * rate)
        new_idx = np.linspace(0, T-1, new_T)
        
        imu2 = np.zeros_like(x)
        for c in range(x.shape[1]):
            imu2[:, c] = np.interp(old_idx, new_idx, np.interp(new_idx, old_idx, x[:, c]))

        return imu2


class Shift:
    def __init__(self, max_shift=0.3):
        """Shift sequence forward or backward by at most max_shift * T timesteps.
        
        Args:
            max_shift: Maximum shift as fraction of sequence length (default: 0.3 for 30%)
        """
        self.max_shift = max_shift

    def __call__(self, x):
        T = x.shape[0]
        max_shift_steps = int(T * self.max_shift)
        
        # Random shift amount (negative for backward, positive for forward)
        shift = np.random.randint(-max_shift_steps, max_shift_steps + 1)
        
        if shift == 0:
            return x
        
        shifted = np.zeros_like(x)
        
        if shift > 0:  # Forward shift
            # Copy shifted data, truncate the end
            shifted[shift:] = x[:T-shift]
            # Pad the beginning with zeros (or could use first value)
            shifted[:shift] = 0  # Zero padding
        else:  # Backward shift
            # Copy shifted data, truncate the beginning
            shifted[:T+shift] = x[-shift:]
            # Pad the end with zeros (or could use last value)
            shifted[T+shift:] = 0  # Zero padding
        
        return shifted




def get_imu_augmentation():
    return Compose([
        Jitter(sigma=0.03),
        Scaling(sigma=0.1),
        RandomCrop(crop_ratio=0.9),
        TimeWarp(sigma=0.2),
        TimeStretch(rate_range=(0.8, 1.2)),
        Shift(max_shift=0.3),
        ChannelDropout(drop_prob=0.05)
    ])
