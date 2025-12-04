import torch
import torch.nn as nn
from imu_ssl.loss.info_nce import info_nce
from imu_ssl.models.projection_head import ProjectionHead

class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        D = encoder.network[-1].conv1.out_channels  # last TCN hidden dim

        # SimCLR projection head
        self.projector = ProjectionHead(D, D, proj_dim)

    def forward(self, x1, x2):
        # x1, x2: (B, T, C)
        h1 = self.encoder(x1)  # (B, T, D)
        h2 = self.encoder(x2)

        # SimCLR loss
        # global pooling -> get (B, D)
        z1 = h1.mean(dim=1)
        z2 = h2.mean(dim=1)

        # projection head MLP
        z1 = self.projector(z1)  
        z2 = self.projector(z2)

        loss = info_nce(z1, z2)
        return loss
