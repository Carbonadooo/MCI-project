import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from pathlib import Path

from imu_ssl.datasets.data_index import DataIndex
from imu_ssl.datasets.imu_dataset import build_dataloader
from imu_ssl.datasets.imu_augmentation import (
    Compose, Jitter, Scaling, RandomCrop, TimeWarp, ChannelDropout
)
from imu_ssl.models.tcn_encoder import TCNEncoder
from imu_ssl.models.ts_tcc_model import TSTCC


# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA = DATA_DIR / "metadata.json"

BATCH_SIZE = 8
SEQ_LEN = 1440
EPOCHS = 20
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = PROJECT_ROOT / "checkpoints"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Augmentation (light version, more stable)
# --------------------------------------------------
def get_augmentation():
    return Compose([
        Jitter(0.01),
        Scaling(0.05),
        # RandomCrop(0.75),
        TimeWarp(0.2),
        # ChannelDropout(0.05),
    ])


# --------------------------------------------------
# Train Loop
# --------------------------------------------------
def main():

    print("Loading metadata:", METADATA)
    index = DataIndex(METADATA)

    # split
    train_files, test_files = index.train_test_split(ratio=0.8)
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")

    # loader
    train_loader = build_dataloader(
        train_files,
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_seq_len=SEQ_LEN,
    )

    # model
    encoder = TCNEncoder(input_ch=9)
    model = TSTCC(encoder).to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # optional scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    augment = get_augmentation()

    best_loss = float("inf")

    # ----------------------------------------
    # Training
    # ----------------------------------------
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        model.train()
        epoch_loss_sum = 0.0
        epoch_ctx_sum = 0.0
        epoch_tmp_sum = 0.0
        epoch_steps = 0

        for imu_batch in train_loader:      # imu_batch: (B, T, 9)
            imu_batch = imu_batch.numpy()   # convert to np for augment

            x1_list, x2_list = [], []
            for seq in imu_batch:
                x1_list.append(torch.tensor(augment(seq), dtype=torch.float32))
                x2_list.append(torch.tensor(augment(seq), dtype=torch.float32))

            x1 = torch.stack(x1_list).to(DEVICE)
            x2 = torch.stack(x2_list).to(DEVICE)

            optimizer.zero_grad()
            loss, stats = model(x1, x2)
            loss.backward()

            # gradient clipping (very helpful for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_ctx_sum += stats["context"]
            epoch_tmp_sum += stats["temporal"]
            epoch_steps += 1

        avg_loss = epoch_loss_sum / epoch_steps
        avg_ctx = epoch_ctx_sum / epoch_steps
        avg_tmp = epoch_tmp_sum / epoch_steps

        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}  "
              f"(ctx={avg_ctx:.4f}, tmp={avg_tmp:.4f})")

        # update scheduler
        scheduler.step()

        # save latest checkpoint (overwrite)
        torch.save(model.state_dict(), OUT_DIR / "latest.pt")

        # save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUT_DIR / "best.pt")
            print(f"✓ Updated best model (loss={best_loss:.4f})")

    print("\nTraining complete. Best loss =", best_loss)


if __name__ == "__main__":
    main()
