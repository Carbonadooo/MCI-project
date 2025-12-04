#!/usr/bin/env python3
"""
Retrieval evaluation for TS-TCC learned IMU encoder
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

# Your code imports
from imu_ssl.datasets.data_index import DataIndex
from imu_ssl.datasets.imu_dataset import IMUDataset
from imu_ssl.inference.tstcc_inference import TSTCCInference


# -------------------------------------------------------------------
# 1. Extract embeddings for all IMU files
# -------------------------------------------------------------------
def extract_embeddings(model, file_list):
    embeddings = []
    labels = []

    print(f"\nEncoding {len(file_list)} files...\n")

    for path in tqdm(file_list):
        imu, _ = IMUDataset._load_hdf5(IMUDataset, path)
        imu = np.asarray(imu, dtype=np.float32)
        emb = model.encode(imu)       # (D,)
        embeddings.append(emb)

    import os
    # labels = folder name (activity)
    for p in file_list:
        # labels.append(p.split("/")[-2])   # folder = activity


        activity = os.path.basename(os.path.dirname(p))
        labels.append(activity)


    return np.vstack(embeddings), np.array(labels)


# -------------------------------------------------------------------
# 2. Retrieval metrics
# -------------------------------------------------------------------
def retrieval_topk(embeddings, labels, k=5):
    N = len(embeddings)
    embeds = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    correct = 0
    for i in range(N):
        query = embeds[i]
        sims = embeds @ query
        sims[i] = -1  # remove self
        topk_idx = np.argsort(-sims)[:k]
        if (labels[topk_idx] == labels[i]).any():
            correct += 1

    return correct / N


def compute_map(embeddings, labels):
    embeds = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    N = len(embeds)
    APs = []

    for i in range(N):
        q = embeds[i]
        sims = embeds @ q
        sims[i] = -1
        ranked = np.argsort(-sims)
        ranked_labels = labels[ranked]

        relevant = (ranked_labels == labels[i])
        rel_pos = np.where(relevant)[0]

        if len(rel_pos) == 0:
            APs.append(0.0)
            continue

        precisions = []
        for rank_idx, pos in enumerate(rel_pos):
            precisions.append((rank_idx + 1) / (pos + 1))
        APs.append(np.mean(precisions))

    return np.mean(APs)


# -------------------------------------------------------------------
# 3. Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True,
                        help="JSON metadata path for DataIndex")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="TS-TCC encoder checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print("\n===================================================")
    print("  TS-TCC Retrieval Evaluation")
    print("===================================================\n")

    # --------------------- Load data ---------------------
    index = DataIndex(args.metadata)  # metadata: activity -> file list
    file_list = index.all_file_list()

    print(f"Total files: {len(file_list)}")
    print("Activities:", index.get_activity_names(), "\n")

    # --------------------- Load model ---------------------
    model = TSTCCInference(args.checkpoint, device=args.device)

    # --------------------- Extract embeddings ---------------------
    embeddings, labels = extract_embeddings(model, file_list)

    # --------------------- Retrieval metrics ---------------------
    r_at_k = retrieval_topk(embeddings, labels, k=args.k)
    map_score = compute_map(embeddings, labels)

    print("\n===================================================")
    print("  RETRIEVAL RESULTS")
    print("===================================================")
    print(f"Recall@{args.k}: {r_at_k:.4f} ({r_at_k*100:.2f}%)")
    print(f"mAP:          {map_score:.4f} ({map_score*100:.2f}%)")
    print("===================================================\n")


if __name__ == "__main__":
    main()
