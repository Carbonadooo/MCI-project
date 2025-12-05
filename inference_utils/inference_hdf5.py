#!/usr/bin/env python3
"""
Simple script to load HDF5 files and run IMU2CLIP inference
Specifically designed for your data format
"""

import numpy as np
import h5py
from .inference_imu2clip import IMU2CLIPInference
from argparse import ArgumentParser


def resample_imu(signal, timestamps, target_rate=200):
    """
    Resample IMU signal to target sampling rate
    
    Args:
        signal: (6, N) numpy array
        timestamps: (N,) numpy array in seconds
        target_rate: target sampling rate in Hz
        
    Returns:
        resampled signal (6, M) and timestamps (M,)
    """
    import torch
    import torchaudio.functional as F
    
    # Calculate current sampling rate
    duration = timestamps[-1] - timestamps[0]
    current_rate = len(timestamps) / duration
    
    if abs(current_rate - target_rate) < 1.0:  # Close enough
        return signal, timestamps
    
    # Resample each channel
    signal_tensor = torch.FloatTensor(signal)
    resampled = F.resample(
        signal_tensor,
        orig_freq=int(current_rate),
        new_freq=target_rate
    )
    
    # Generate new timestamps
    n_samples = resampled.shape[1]
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], n_samples)
    
    return resampled.numpy(), new_timestamps


def load_hdf5_imu(hdf5_path, verbose=True, resample_to_200hz=True, num_channels=6):
    """
    Load IMU data from HDF5 file
    
    Expected HDF5 structure:
        /imu/data - IMU sensor data
        /imu/timestamps - timestamps
    
    Args:
        hdf5_path: Path to .hdf5 file
        verbose: Print loading information
        resample_to_200hz: Resample to 200 Hz (model's expected rate)
        num_channels: Number of channels to load (6, 9, 18, or 21)
                      6 = accel + gyro
                      9 = accel + gyro + magnetometer
                      18/21 = loads 9 channels (engineered features added later by inference)
        
    Returns:
        dict with 'imu_data' (num_channels or 9, N) and 'timestamps' (N,)
        Note: For num_channels=18 or 21, returns 9 channels (features computed separately)
    """
    if verbose:
        print(f"Loading: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as hf:
        # Load IMU data
        imu_data = hf['imu/data'][:]
        imu_timestamps = hf['imu/timestamps'][:]
        
        if verbose:
            print(f"  Raw IMU shape: {imu_data.shape}")
            print(f"  Timestamps: {len(imu_timestamps)} samples")
    
    # Convert to (channels, N) format
    if imu_data.shape[0] > imu_data.shape[1]:
        # Shape is (N, channels), transpose to (channels, N)
        imu_data = imu_data.T
        if verbose:
            print(f"  Transposed to: {imu_data.shape}")
    
    # Extract specified number of channels
    if num_channels == 6:
        # 6 channels: accel_xyz + gyro_xyz
        if imu_data.shape[0] > 6:
            if verbose:
                print(f"  Extracting first 6 channels (found {imu_data.shape[0]})")
            imu_data = imu_data[:6, :]
        elif imu_data.shape[0] < 6:
            raise ValueError(f"Expected at least 6 IMU channels, got {imu_data.shape[0]}")
    elif num_channels in (9, 18, 21):
        # 9 channels: accel_xyz + gyro_xyz + mag_xyz
        # (For 18/21-channel mode, we load 9 channels; engineered features added by inference)
        if imu_data.shape[0] > 9:
            if verbose:
                channel_msg = "9 channels" if num_channels == 9 else "9 channels (engineered mode)"
                print(f"  Extracting first {channel_msg} (found {imu_data.shape[0]})")
            imu_data = imu_data[:9, :]
        elif imu_data.shape[0] < 9:
            raise ValueError(f"Expected at least 9 IMU channels, got {imu_data.shape[0]}")
    else:
        raise ValueError(f"num_channels must be 6, 9, 18, or 21, got {num_channels}")
    
    # Calculate original sampling rate
    duration = (imu_timestamps[-1] - imu_timestamps[0])
    original_rate = len(imu_timestamps) / duration
    
    if verbose:
        print(f"  Original shape: {imu_data.shape}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Original sampling rate: {original_rate:.1f} Hz")
    
    # Resample to 200 Hz if needed (model's expected rate)
    if resample_to_200hz and abs(original_rate - 200.0) > 1.0:
        imu_data, imu_timestamps = resample_imu(imu_data, imu_timestamps, target_rate=200)
        if verbose:
            print(f"  Resampled to 200 Hz: {imu_data.shape}")
    
    return {
        'imu_data': imu_data,
        'timestamps': imu_timestamps,
        'sampling_rate': 200.0 if resample_to_200hz else original_rate
    }


def encode_with_windows(model, imu_data, window_size=1000, stride=50):
    """
    Encode IMU data using sliding windows and average embeddings
    Better for comparing long sequences
    
    Args:
        model: IMU2CLIPInference instance
        imu_data: numpy array (6, N)
        window_size: samples per window (1000 = 5 sec at 200 Hz)
        stride: stride between windows (500 = 2.5 sec, 50% overlap)
        
    Returns:
        averaged embedding (512,)
    """
    n_samples = imu_data.shape[1]
    
    # If data is shorter than window, just use single encoding
    if n_samples <= window_size:
        emb = model.encode(imu_data)
        return emb
    
    # Extract sliding windows and encode each
    window_embeddings = []
    
    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = imu_data[:, start_idx:end_idx]
        
        # Encode window
        window_emb = model.encode(window)
        window_embeddings.append(window_emb)

    # Drop out the windows outside of 2 sigma of the mean (outlier removal)
    window_embeddings = np.array(window_embeddings)
    
    # Calculate initial mean
    mean_emb = np.mean(window_embeddings, axis=0)
    
    # Calculate L2 distance of each window from the mean
    distances = np.linalg.norm(window_embeddings - mean_emb, axis=1)
    
    # Calculate mean and std of distances
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Keep only windows within 1 sigma
    mask = distances <= (mean_dist + 2 * std_dist)
    filtered_embeddings = window_embeddings[mask]
    
    # If too many filtered out, keep at least half
    if len(filtered_embeddings) < len(window_embeddings) // 2:
        # Sort by distance and keep closest half
        sorted_indices = np.argsort(distances)
        n_keep = len(window_embeddings) // 2
        filtered_embeddings = window_embeddings[sorted_indices[:n_keep]]
    
    # Average filtered window embeddings
    avg_embedding = np.mean(filtered_embeddings, axis=0)
    
    return avg_embedding


def match_action_sliding_window(model, imu_data, action_classes, window_size=1000, stride=100):
    """
    Match IMU data to action classes using sliding windows and voting
    
    Args:
        model: IMU2CLIPInference instance
        imu_data: numpy array (6, N) - should be at 200 Hz
        action_classes: list of action names
        window_size: samples per window (1000 = 5 seconds at 200 Hz)
        stride: stride between windows in samples
        
    Returns:
        dict with predictions, scores, and per-window results
    """
    import clip
    import torch
    
    device = model.device
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Encode text descriptions once
    text_tokens = clip.tokenize(action_classes).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings = text_embeddings.cpu().numpy()
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    n_samples = imu_data.shape[1]
    
    # If data is shorter than window, just use single prediction
    if n_samples <= window_size:
        imu_embedding = model.encode(imu_data)
        imu_embedding = imu_embedding / np.linalg.norm(imu_embedding)
        similarities = text_embeddings @ imu_embedding
        best_idx = np.argmax(similarities)
        
        return {
            'predicted_action': action_classes[best_idx],
            'confidence': float(similarities[best_idx]),
            'all_scores': {action: float(score) for action, score in zip(action_classes, similarities)},
            'n_windows': 1,
            'voting_used': False
        }
    
    # Extract sliding windows
    window_predictions = []
    window_embeddings = []
    
    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = imu_data[:, start_idx:end_idx]
        
        # Encode window
        window_emb = model.encode(window)
        window_emb = window_emb / np.linalg.norm(window_emb)
        window_embeddings.append(window_emb)
        
        # Compute similarities for this window
        window_sims = text_embeddings @ window_emb
        window_pred = action_classes[np.argmax(window_sims)]
        window_predictions.append(window_pred)
    
    # Voting: count predictions
    from collections import Counter
    vote_counts = Counter(window_predictions)
    voted_action = vote_counts.most_common(1)[0][0]
    
    # Aggregate scores across all windows
    avg_embedding = np.mean(window_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    avg_similarities = text_embeddings @ avg_embedding
    
    return {
        'predicted_action': voted_action,
        'confidence': float(avg_similarities[action_classes.index(voted_action)]),
        'all_scores': {action: float(score) for action, score in zip(action_classes, avg_similarities)},
        'n_windows': len(window_predictions),
        'voting_used': True,
        'vote_counts': dict(vote_counts),
        'window_predictions': window_predictions
    }


def match_action(model, imu_data, action_classes, use_sliding_window=True):
    """
    Match IMU data to predefined action classes
    
    Args:
        model: IMU2CLIPInference instance
        imu_data: numpy array (6, N) - should be at 200 Hz
        action_classes: list of action names
        use_sliding_window: Use sliding windows for long sequences
        
    Returns:
        dict with predictions and scores
    """
    if use_sliding_window and imu_data.shape[1] > 1000:
        return match_action_sliding_window(model, imu_data, action_classes)
    
    # Single window prediction (original method)
    import clip
    import torch
    
    # Encode IMU
    imu_embedding = model.encode(imu_data)
    imu_embedding = imu_embedding / np.linalg.norm(imu_embedding)
    
    # Encode text descriptions with CLIP
    device = model.device
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    text_tokens = clip.tokenize(action_classes).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings = text_embeddings.cpu().numpy()
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = text_embeddings @ imu_embedding
    
    # Get best match
    best_idx = np.argmax(similarities)
    
    return {
        'predicted_action': action_classes[best_idx],
        'confidence': float(similarities[best_idx]),
        'all_scores': {action: float(score) for action, score in zip(action_classes, similarities)},
        'voting_used': False,
        'n_windows': 1
    }


def main():
    parser = ArgumentParser(description="IMU2CLIP inference for HDF5 files")
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to HDF5 file (e.g., /path/to/clap_once_20251108_191951.hdf5)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./saved/i2c/i2c_s_i_t_t_se_mw2_w_2.5_master-epoch=01-val_loss=5.87.ckpt",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda'],
        help="Device to use"
    )
    parser.add_argument(
        "--actions",
        type=str,
        nargs='+',
        default=["clap once", "cut vegetables with knife", "shake bottle", "punch forward", "unplug usb cable", "screw bottle cap"],
        help="Action classes to match against"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save embedding to .npy file"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs='*',
        default=None,
        help="Additional HDF5 files to compare against"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("IMU2CLIP HDF5 Inference")
    print("=" * 70)
    print()
    
    # Load model FIRST
    print("Loading model...")
    model = IMU2CLIPInference(args.checkpoint, device=args.device)
    print()
    
    # Load HDF5 file
    print("Loading HDF5 data...")
    data = load_hdf5_imu(args.hdf5_file)
    imu_data = data['imu_data']
    print()
    
    # Encode to CLIP embedding (using sliding windows for long sequences)
    print("Encoding IMU to CLIP space...")
    if imu_data.shape[1] > 1000:
        n_windows = (imu_data.shape[1] - 1000) // 500 + 1
        print(f"  Using {n_windows} sliding windows (sequence > 5 sec)")
        embedding = encode_with_windows(model, imu_data)
    else:
        embedding = model.encode(imu_data)
    
    print(f"✓ Generated embedding: {embedding.shape}")
    print(f"  Norm: {np.linalg.norm(embedding):.4f}")
    print()
    
    # Compare with other HDF5 files (if specified)
    compare_files = args.compare if args.compare else []
    
    # Add default comparison files if none specified
    if not compare_files:
        compare_files = [
            "/home/chuye/Documents/MCI-project/data/nov2_set/clap_once/clap_once_20251108_221734.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/drink_water/drink_water_20251108_210801.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/screw_bottle_cap/screw_bottle_cap_20251108_204604.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/screw_bottle_cap/screw_bottle_cap_20251116_165848.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/clap_once/clap_once_20251116_165201.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/clap_once/clap_once_20251116_165218.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/screw_bottle_cap/screw_bottle_cap_20251108_223420.hdf5",
            "/home/chuye/Documents/MCI-project/data/nov2_set/twist_towel/twist_towel_20251108_205207.hdf5"
        ]
    
    if compare_files:
        print("=" * 70)
        print("Comparing to other IMU files (L2 distance in CLIP space)")
        print("Using sliding window embeddings for robust comparison")
        print("=" * 70)
        
        # Encode main file with sliding windows if needed
        main_emb = encode_with_windows(model, imu_data)
        main_emb = main_emb / np.linalg.norm(main_emb)
        
        compare_results = []
        
        print(f"\nComparing against {len(compare_files)} files...")
        for compare_path in compare_files:
            try:
                # Load and encode comparison file
                comp_data = load_hdf5_imu(compare_path, verbose=False)
                comp_imu = comp_data['imu_data']
                
                # Use sliding windows for encoding
                comp_emb = encode_with_windows(model, comp_imu)
                comp_emb = comp_emb / np.linalg.norm(comp_emb)
                
                # Compute L2 distance
                l2_dist = np.linalg.norm(main_emb - comp_emb)
                
                # Also compute cosine similarity
                cosine_sim = np.dot(main_emb, comp_emb)
                
                compare_results.append({
                    'path': compare_path,
                    'filename': compare_path.split('/')[-1],
                    'l2_distance': l2_dist,
                    'cosine_similarity': cosine_sim,
                    'duration': (comp_data['timestamps'][-1] - comp_data['timestamps'][0])
                })
            except Exception as e:
                print(f"⚠ Failed to compare: {compare_path.split('/')[-1]}")
                print(f"  Error: {e}")
        
        # Display comparison results
        if compare_results:
            print("\nResults (sorted by similarity):")
            compare_results.sort(key=lambda x: x['l2_distance'])
            
            for i, result in enumerate(compare_results):
                filename = result['filename']
                l2_dist = result['l2_distance']
                cos_sim = result['cosine_similarity']
                dur = result['duration']
                
                # Visual bar for similarity
                bar_length = int(cos_sim * 50)
                bar = '█' * bar_length
                
                print(f"{i+1}. {filename} ({dur:.1f}s)")
                print(f"   L2 distance: {l2_dist:.4f}  |  Cosine sim: {cos_sim:.4f}")
                print(f"   {bar}")
                print()
        else:
            print("No comparison files could be processed.")
        
        print("=" * 70)
        print()
    
    # Match to actions
    print("Matching to action classes...")
    print(f"  Actions: {args.actions}")
    results = match_action(model, imu_data, args.actions)
    print()
    
    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Predicted Action: {results['predicted_action']}")
    print(f"Confidence: {results['confidence']:.4f}")
    
    # Show voting information if used
    if results.get('voting_used', False):
        print(f"Method: Sliding Window Voting ({results['n_windows']} windows)")
        print("\nWindow Voting Results:")
        for action, count in sorted(results['vote_counts'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / results['n_windows']) * 100
            bar = '█' * int(pct / 2)  # Scale to 50 chars max
            print(f"  {count:3d}/{results['n_windows']:3d} ({pct:5.1f}%) {bar:50s} {action}")
    else:
        print(f"Method: Single Window")
    print()
    
    print("Aggregated Scores:")
    sorted_scores = sorted(results['all_scores'].items(), key=lambda x: x[1], reverse=True)
    for action, score in sorted_scores:
        bar = '█' * int(score * 50) if score > 0 else ''
        marker = '→' if action == results['predicted_action'] else ' '
        print(f"{marker} {score:6.4f} {bar:50s} {action}")
    print()
    
    # Save embedding if requested
    if args.output:
        np.save(args.output, embedding)
        print(f"✓ Saved embedding to: {args.output}")
        print()
    
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()

