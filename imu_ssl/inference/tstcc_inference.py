import torch
import numpy as np
from imu_ssl.models.tcn_encoder import TCNEncoder
from imu_ssl.models.ts_tcc_model import TSTCC


class TSTCCInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device

        # 1. Load encoder
        encoder = TCNEncoder(input_ch=9)
        model = TSTCC(encoder)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        self.encoder = model.encoder.to(device)  # only need encoder for inference

    @torch.no_grad()
    def encode(self, imu_np):
        """
        imu_np: (T, 9) numpy array
        return: (D,) embedding vector
        """
        x = torch.tensor(imu_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T, 9)
        h_seq = self.encoder(x)  # (1, T, D)
        h = h_seq.mean(dim=1).squeeze(0)  # (D,)
        h = h / torch.norm(h)
        return h.cpu().numpy()  # (D,)
