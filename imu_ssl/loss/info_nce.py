import torch
import torch.nn.functional as F

def info_nce(z1, z2, temperature=0.1):
    """
    z1, z2: (B, D)
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(B, device=z1.device)

    return F.cross_entropy(logits, labels)
