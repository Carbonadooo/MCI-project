import torch
import torch.nn as nn
from imu_ssl.loss.info_nce import info_nce
class TemporalContrast(nn.Module):
    def __init__(self, proj_head):
        super().__init__()
        self.proj = proj_head

    def forward(self, h1_seq, h2_seq):
        # h1_seq: (B, T, D)
        # h2_seq: (B, T, D)
        B, T, D = h1_seq.size()

        # flatten time dimension for projection
        z1 = self.proj(h1_seq.reshape(B*T, D))
        z2 = self.proj(h2_seq.reshape(B*T, D))

        z1 = z1.reshape(B, T, -1)
        z2 = z2.reshape(B, T, -1)

        # compute InfoNCE across batch+time
        # reshape to (B*T, D)
        z1_flat = z1.reshape(B*T, -1)
        z2_flat = z2.reshape(B*T, -1)

        return info_nce(z1_flat, z2_flat)
