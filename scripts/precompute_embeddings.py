#!/usr/bin/env python3
"""
Precompute IMU2CLIP embeddings for all HDF5 files under a data root.

- Scans recursively for *.hdf5 files under --data_root (default: PROJECT_ROOT/data/all)
- Loads IMU (resampled to 200 Hz, 9 raw channels) and computes 21-ch features via IMU2CLIPInference
- Aggregates with the same sliding-window logic as CLI (encode_with_windows)
- Saves a normalized embedding sidecar next to each HDF5: <file>.imu2clip.npy

Usage:
  python scripts/precompute_embeddings.py \
    --data_root /path/to/data/all \
    --checkpoint models/best_model.ckpt \
    --model_name MW2StackRNNPooling21Ch \
    --num_channels 21 \
    [--overwrite]
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure project root in path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_utils.inference_imu2clip import IMU2CLIPInference
from inference_utils.inference_hdf5 import load_hdf5_imu, encode_with_windows

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "all"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ever" / "best_model.ckpt"


def parse_args():
    p = argparse.ArgumentParser(description="Precompute IMU2CLIP embeddings for HDF5 dataset")
    p.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Root directory containing HDF5 files (recursively)")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Checkpoint path for IMU2CLIP model")
    p.add_argument("--model_name", type=str, default="MW2StackRNNPooling21Ch", help="Model class name")
    p.add_argument("--num_channels", type=int, default=21, help="Model input channels (21 for raw+rotation+world-acc)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .imu2clip.npy sidecars")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}")
        sys.exit(1)

    # Initialize inference
    print(f"[INFO] Loading model: {args.model_name} from {args.checkpoint}")
    infer = IMU2CLIPInference(
        checkpoint_path=args.checkpoint,
        device="cpu",
        model_name=args.model_name,
        num_channels=args.num_channels,
    )

    # Discover all .hdf5 files
    h5_files = sorted(list(data_root.rglob("*.hdf5")))
    print(f"[INFO] Found {len(h5_files)} HDF5 files under {data_root}")

    processed = 0
    skipped = 0
    failed = 0

    for path in h5_files:
        sidecar = Path(str(path) + ".imu2clip.npy")
        if sidecar.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            # Load IMU 9-ch and compute windows embedding
            data = load_hdf5_imu(str(path), verbose=False, resample_to_200hz=True, num_channels=9)
            emb = encode_with_windows(infer, data['imu_data'])
            # Normalize (cosine similarity expects unit norm)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            # Save sidecar
            np.save(sidecar, emb)
            processed += 1
            if processed % 50 == 0:
                print(f"[INFO] Processed {processed} files...")
        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")
            failed += 1

    print(f"[DONE] Processed={processed}, Skipped={skipped}, Failed={failed}")


if __name__ == "__main__":
    main()