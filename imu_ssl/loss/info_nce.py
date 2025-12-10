import torch
import torch.nn.functional as F

def info_nce(z1, z2, temperature=0.1, neg_weight=1.0):
    batch_size = z1.shape[0]

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # positive logits
    pos = torch.sum(z1 * z2, dim=1) / temperature  # (B,)

    # all negatives: z1 vs z2_j for j != i
    neg = z1 @ z2.T  # (B, B)
    mask = torch.eye(batch_size, device=z1.device).bool()
    neg = neg.masked_fill(mask, -9e15) / temperature

    neg = neg * neg_weight

    logits = torch.cat([pos.unsqueeze(1), neg], dim=1) # (B, 1+B)

    labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)

    loss = F.cross_entropy(logits, labels)
    return loss
