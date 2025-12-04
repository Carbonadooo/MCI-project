import torch
import torch.nn as nn
from imu_ssl.loss.info_nce import info_nce
class ContextualContrast(nn.Module):
    def __init__(self, proj_head):
        super().__init__()
        self.proj = proj_head

    def forward(self, h1_seq, h2_seq):
        # mean pool over time
        h1 = h1_seq.mean(dim=1)    # (B, D)
        h2 = h2_seq.mean(dim=1)

        z1 = self.proj(h1)
        z2 = self.proj(h2)

        return info_nce(z1, z2)
