import sys
import os
import time
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
from imu_ssl.models.SimCLR_model import SimCLR


# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA = DATA_DIR / "metadata_train.json"

BATCH_SIZE = 32
SEQ_LEN = 1440
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
OUT_DIR = PROJECT_ROOT / "checkpoints"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Augmentation (light version, more stable)
# --------------------------------------------------
def get_augmentation():
    return Compose([
        Jitter(0.01),
        Scaling(0.05),
        RandomCrop(0.9),
        TimeWarp(0.2),
        # ChannelDropout(0.05),
    ])


# --------------------------------------------------
# Train Loop
# --------------------------------------------------
def main():

    # training files
    print("Loading metadata:", METADATA)
    index = DataIndex(METADATA)
    train_files = index.all_file_list()
    print("Train samples:", len(train_files))

    # loader
    augment = get_augmentation()
    train_loader = build_dataloader(
        train_files,
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_seq_len=SEQ_LEN,
        augment=augment,
    )

    # model
    encoder = TCNEncoder(input_ch=9)
    model = SimCLR(encoder).to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # optional scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_loss = float("inf")

    # ----------------------------------------
    # Training
    # ----------------------------------------
    print(f"\nStarting training for {EPOCHS} epochs...")
    training_start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====\n")
        epoch_start_time = time.time()

        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for x1, x2 in train_loader:
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_steps += 1

        avg_loss = epoch_loss_sum / epoch_steps
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}  "
              f"Time: {epoch_duration:.2f}s")

        scheduler.step()

        torch.save(model.state_dict(), OUT_DIR / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUT_DIR / "best.pt")
            print(f"✓ Updated best model (loss={best_loss:.4f})")

    total_training_time = time.time() - training_start_time
    print(f"\nTraining complete. Best loss = {best_loss:.4f}")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
