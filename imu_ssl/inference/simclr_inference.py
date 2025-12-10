import torch
import numpy as np
from imu_ssl.models.tcn_encoder import TCNEncoder
from imu_ssl.models.SimCLR_model import SimCLR


class SimCLRInference:
    """
    Load a trained SimCLR model and extract encoder-only embeddings.
    """

    def __init__(self, input_ch, checkpoint_path, device="cuda"):
        self.device = device

        # 1. Build encoder + SimCLR model
        encoder = TCNEncoder(input_ch=input_ch)
        model = SimCLR(encoder)   # contains encoder + projector

        # 2. Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # 3. Keep only encoder for inference
        self.encoder = model.encoder.to(device)
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, imu_np):
        """
        imu_np: (T, 9) numpy array
        return: (D,) feature embedding vector
        """
        # shape: (1, T, 9)
        x = torch.tensor(imu_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        # (1, T, D)
        h_seq = self.encoder(x)

        # mean-pool across time → (1, D)
        h = h_seq.mean(dim=1).squeeze(0)

        # L2 normalization
        h = h / torch.norm(h, p=2)

        return h.cpu().numpy()
