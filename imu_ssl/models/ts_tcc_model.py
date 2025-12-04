import torch
import torch.nn as nn
from imu_ssl.loss.contextual_loss import ContextualContrast
from imu_ssl.loss.temporal_loss import TemporalContrast
from imu_ssl.models.projection_head import ProjectionHead

class TSTCC(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        D = encoder.network[-1].conv1.out_channels  # last TCN hidden dim

        # two heads
        self.context_head = ProjectionHead(D, D, proj_dim)
        self.temporal_head = ProjectionHead(D, D, proj_dim)

        self.context_loss = ContextualContrast(self.context_head)
        self.temporal_loss = TemporalContrast(self.temporal_head)

    def forward(self, x1, x2, lambda_temp=1.0):
        # x1, x2: (B, T, C)

        h1 = self.encoder(x1)  # (B, T, D)
        h2 = self.encoder(x2)

        loss_ctx = self.context_loss(h1, h2)
        loss_temp = self.temporal_loss(h1, h2)

        return loss_ctx + lambda_temp * loss_temp, \
               {"context": loss_ctx.item(), "temporal": loss_temp.item()}
