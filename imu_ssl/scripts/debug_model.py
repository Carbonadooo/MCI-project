import torch
from imu_ssl.models.ts_tcc_model import TSTCC
from imu_ssl.models.tcn_encoder import TCNEncoder

if __name__ == "__main__":
    B, T, C = 8, 200, 9
    x1 = torch.randn(B, T, C)
    x2 = torch.randn(B, T, C)

    encoder = TCNEncoder()
    model = TSTCC(encoder)

    loss, stats = model(x1, x2)
    print("loss:", loss.item(), stats)

