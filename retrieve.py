#!/usr/bin/env python3
"""
IMU-to-IMU Retrieval Evaluation

Evaluates IMU2CLIP embeddings using standard information retrieval metrics:
- Recall@k: What fraction of relevant items are in top-k?
- Precision@k: What fraction of top-k are relevant?
- mAP (Mean Average Precision): Overall retrieval quality

Protocol:
- Leave-one-out: Each sample is used as a query once
- Gallery: All other samples (N-1 per query)
- Relevance: Same action class = relevant

Usage:
    # 21-channel model (default here)
    python retrive.py --val_data_dir /path/to/data_dir \
        --checkpoint /Users/jianuoqiu/Documents/GT/CS8803/MCI-project/models/best_model.ckpt \
        --model_name MW2StackRNNPooling21Ch --num_channels 21
    
    # 6-channel model
    python retrive.py --val_data_dir /path/to/data_dir
    
    # 9-channel model
    python retrive.py --val_data_dir /path/to/data_dir \
        --checkpoint log/finetuned_models_9ch/best_model.ckpt \
        --model_name MW2StackRNNPooling9Ch \
        --num_channels 9
    
    # 18-channel model
    python retrive.py --val_data_dir /path/to/data_dir \
        --checkpoint log/finetuned_models_18ch/best_model.ckpt \
        --model_name MW2StackRNNPooling18Ch \
        --num_channels 18
    
    # Custom k values
    python retrive.py --val_data_dir /path/to/data_dir \
        --k_values 1 5 10 50 --device cuda

See RETRIEVAL_EVALUATION.md for detailed documentation.
"""
import os
import glob
import h5py
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from inference_utils.inference_imu2clip import IMU2CLIPInference
from inference_utils.inference_hdf5 import load_hdf5_imu, encode_with_windows
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(data_dir, model, max_per_class=None, num_channels=6):
    """Load all HDF5 files and encode to embeddings"""
    embeddings = []
    labels = []
    filenames = []
    
    # Scan directories
    class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    
    print(f"Found {len(class_dirs)} classes")
    if num_channels == 18:
        print(f"Using {num_channels}-channel mode (9 IMU + 9 rotation features)")
    elif num_channels == 21:
        print(f"Using {num_channels}-channel mode (9 IMU + 9 rotation + 3 world-frame accel)")
    else:
        print(f"Using {num_channels}-channel IMU data")

    
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        hdf5_files = sorted(glob.glob(str(class_dir / "*.hdf5")))
        
        if max_per_class:
            hdf5_files = hdf5_files[:max_per_class]
        
        if len(hdf5_files) < 2:
            print(f"  {class_name}: {len(hdf5_files)} samples - SKIPPED (need at least 2)")
            continue
        
        print(f"  {class_name}: {len(hdf5_files)} samples", end="")
        
        for hdf5_file in hdf5_files:
            try:
                # Load IMU data
                # For 18/21-channel modes: load 9 channels (engineered features computed automatically)
                load_channels = 9 if num_channels in (18, 21) else num_channels
                data = load_hdf5_imu(hdf5_file, verbose=False, num_channels=load_channels)
                imu_data = data['imu_data']  # Shape: (6, N), (9, N)
                
                # Encode using sliding windows
                emb = encode_with_windows(model, imu_data)
                emb = emb / np.linalg.norm(emb)  # Normalize
                
                embeddings.append(emb)
                labels.append(class_name)
                filenames.append(os.path.basename(hdf5_file))
                
            except Exception as e:
                print(f"\n    ⚠ Failed: {os.path.basename(hdf5_file)} - {e}")
        
        print(f" ✓")
    
    return np.array(embeddings), np.array(labels), filenames


def compute_retrieval_metrics(embeddings, labels, k_values=[1, 10, 50]):
    """
    Compute retrieval metrics for IMU-to-IMU retrieval (leave-one-out)
    
    Args:
        embeddings: (N, D) array of normalized embeddings
        labels: (N,) array of class labels
        k_values: list of k values for Recall@k and Precision@k
    
    Returns:
        dict with all metrics
    """
    n_samples = len(embeddings)
    
    # Initialize metric accumulators
    recall_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    average_precisions = []
    
    print(f"\n{'='*70}")
    print(f"Computing retrieval metrics (leave-one-out)")
    print(f"Total queries: {n_samples}")
    print(f"Gallery size per query: {n_samples - 1}")
    print(f"{'='*70}\n")
    
    # Process each sample as a query
    for i in range(n_samples):
        # Query embedding and label
        query_emb = embeddings[i]
        query_label = labels[i]
        
        # Gallery: all samples except query itself (leave-one-out)
        gallery_indices = [j for j in range(n_samples) if j != i]
        gallery_embs = embeddings[gallery_indices]
        gallery_labels = labels[gallery_indices]
        
        # Compute similarities (cosine similarity = dot product for normalized embeddings)
        similarities = gallery_embs @ query_emb
        
        # Rank by similarity (descending order)
        ranked_indices = np.argsort(-similarities)
        ranked_labels = gallery_labels[ranked_indices]
        
        # Identify relevant items (same class as query)
        relevant_mask = (ranked_labels == query_label)
        relevant_positions = np.where(relevant_mask)[0]  # Positions of relevant items in ranked list
        n_relevant = relevant_mask.sum()
        
        # Compute Recall@k and Precision@k for each k
        for k in k_values:
            if k <= len(ranked_labels):
                # Number of relevant items in top-k
                top_k_relevant = relevant_mask[:k].sum()
                
                # Recall@k = (relevant in top-k) / (total relevant in gallery)
                recall = top_k_relevant / n_relevant if n_relevant > 0 else 0.0
                recall_at_k[k].append(recall)
                
                # Precision@k = (relevant in top-k) / k
                precision = top_k_relevant / k
                precision_at_k[k].append(precision)
        
        # Compute Average Precision (AP) for this query
        if n_relevant > 0:
            precisions_at_relevant = []
            for rank_among_relevant, position_in_list in enumerate(relevant_positions):
                # At this position, how many relevant items have we seen?
                # rank_among_relevant is 0-indexed count of relevant items so far
                # position_in_list is the actual position in the ranked list (0-indexed)
                precision_at_pos = (rank_among_relevant + 1) / (position_in_list + 1)
                precisions_at_relevant.append(precision_at_pos)
            
            # Average Precision = mean of precisions at all relevant positions
            ap = np.mean(precisions_at_relevant)
        else:
            ap = 0.0
        
        average_precisions.append(ap)
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  Processed {i + 1}/{n_samples} queries...")
    
    # Compute mean metrics across all queries
    results = {}
    for k in k_values:
        results[f'Recall@{k}'] = np.mean(recall_at_k[k])
        results[f'Precision@{k}'] = np.mean(precision_at_k[k])
    results['mAP'] = np.mean(average_precisions)
    
    return results


def print_results(results):
    """Pretty print retrieval results"""
    print(f"\n{'='*70}")
    print("RETRIEVAL RESULTS")
    print(f"{'='*70}\n")
    
    # Recall@k
    print("Recall@k (how many relevant items are in top-k):")
    for key in sorted(results.keys()):
        if key.startswith('Recall'):
            print(f"  {key:15s}: {results[key]:.4f} ({results[key]*100:.2f}%)")
    
    print()
    
    # Precision@k
    print("Precision@k (how clean is the top-k):")
    for key in sorted(results.keys()):
        if key.startswith('Precision'):
            print(f"  {key:15s}: {results[key]:.4f} ({results[key]*100:.2f}%)")
    
    print()
    
    # mAP
    print(f"mAP (Mean Average Precision):")
    print(f"  mAP: {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="IMU-to-IMU Retrieval Evaluation using IMU2CLIP embeddings"
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="/Users/jianuoqiu/Documents/GT/CS8803/MCI-project/data/validate_set",
        help="Data directory with class folders"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/jianuoqiu/Documents/GT/CS8803/MCI-project/models/best_model.ckpt",
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MW2StackRNNPooling21Ch",
        choices=["MW2StackRNNPooling", "MW2StackRNNPooling9Ch", "MW2StackRNNPooling18Ch", "MW2StackRNNPooling21Ch"],
        help="Model architecture"
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=21,
        choices=[6, 9, 18, 21],
        help="Number of channels: 6 (accel+gyro), 9 (+mag), 18 (+rotation), 21 (+rotation + world-frame accel)"
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Max samples per class (default: all)"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs='+',
        default=[1, 10, 50],
        help="k values for Recall@k and Precision@k (default: 1 10 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda"
    )
    
    args = parser.parse_args()
    
    # Auto-select model name based on num_channels if using default umbrella name
    if args.model_name == "MW2StackRNNPooling":
        if args.num_channels == 9:
            args.model_name = "MW2StackRNNPooling9Ch"
            print(f"ℹ️  Auto-selected: {args.model_name} for {args.num_channels}-channel data\n")
        elif args.num_channels == 18:
            args.model_name = "MW2StackRNNPooling18Ch"
            print(f"ℹ️  Auto-selected: {args.model_name} for {args.num_channels}-channel data (9 IMU + 9 rotation)\n")
        elif args.num_channels == 21:
            args.model_name = "MW2StackRNNPooling21Ch"
            print(f"ℹ️  Auto-selected: {args.model_name} for {args.num_channels}-channel data (9 IMU + 9 rotation + 3 world-frame accel)\n")
    
    print("="*70)
    print("IMU-to-IMU Retrieval Evaluation")
    print("="*70)
    print()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    print(f"Model: {args.model_name} ({args.num_channels}-channel)")
    model = IMU2CLIPInference(
        args.checkpoint, 
        device=args.device,
        model_name=args.model_name,
        num_channels=args.num_channels
    )
    print()
    
    # Load dataset and encode all samples to embeddings
    print(f"Loading and encoding dataset from {args.val_data_dir}")
    embeddings, labels, filenames = load_dataset(
        args.val_data_dir, 
        model, 
        max_per_class=args.max_per_class,
        num_channels=args.num_channels
    )
    
    print(f"\n✓ Loaded {len(embeddings)} samples")
    print(f"✓ Number of classes: {len(np.unique(labels))}")
    
    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} samples")
    
    # Compute retrieval metrics
    results = compute_retrieval_metrics(
        embeddings, 
        labels, 
        k_values=args.k_values
    )
    
    # Print results
    print_results(results)
    
    # Additional analysis: per-class breakdown
    # For each class, compute average mAP when using samples from that class as queries
    print("\nPer-class retrieval performance:")
    print("(Average mAP when using each class as query)\n")
    
    class_aps = {}
    for target_label in unique_labels:
        # Find all queries with this label
        query_indices = np.where(labels == target_label)[0]
        
        if len(query_indices) >= 2:
            label_aps = []
            
            for i in query_indices:
                # Use this sample as query
                query_emb = embeddings[i]
                query_label = labels[i]
                
                # Gallery: all other samples
                gallery_indices = [j for j in range(len(embeddings)) if j != i]
                gallery_embs = embeddings[gallery_indices]
                gallery_labels = labels[gallery_indices]
                
                # Compute similarities and rank
                similarities = gallery_embs @ query_emb
                ranked_indices = np.argsort(-similarities)
                ranked_labels = gallery_labels[ranked_indices]
                
                # Relevant items
                relevant_mask = (ranked_labels == query_label)
                relevant_positions = np.where(relevant_mask)[0]
                n_relevant = relevant_mask.sum()
                
                # Compute AP
                if n_relevant > 0:
                    precisions_at_relevant = []
                    for rank, pos in enumerate(relevant_positions):
                        precision_at_pos = (rank + 1) / (pos + 1)
                        precisions_at_relevant.append(precision_at_pos)
                    ap = np.mean(precisions_at_relevant)
                else:
                    ap = 0.0
                
                label_aps.append(ap)
            
            class_aps[target_label] = np.mean(label_aps)
    
    # Sort by mAP (descending)
    sorted_classes = sorted(class_aps.items(), key=lambda x: x[1], reverse=True)
    for label, ap in sorted_classes:
        print(f"  {label:35s}: {ap:.4f} ({ap*100:.2f}%)")
    
    print(f"\n{'='*70}\n")
    
    
if __name__ == "__main__":
    main()