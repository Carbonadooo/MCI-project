#!/usr/bin/env python3
"""
IMU2CLIP Inference Script
Encode IMU data to CLIP embedding space using trained model
"""

import torch
import numpy as np
from lib.imu_models import MW2StackRNNPooling, MW2StackRNNPooling9Ch, MW2StackRNNPooling18Ch, MW2StackRNNPooling21Ch
from lib.rotation_utils import compute_rotation_features, compute_world_frame_acceleration
from argparse import ArgumentParser


class IMU2CLIPInference:
    """
    Load trained IMU2CLIP model and encode IMU sequences to CLIP embeddings
    """
    
    def __init__(self, checkpoint_path, device='cuda', model_name='MW2StackRNNPooling', num_channels=6):
        """
        Initialize the inference model
        
        Args:
            checkpoint_path: Path to trained .ckpt file
            device: 'cuda' or 'cpu'
            model_name: Model architecture ('MW2StackRNNPooling' or 'MW2StackRNNPooling9Ch')
            num_channels: Number of IMU channels (6 or 9)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_channels = num_channels
        self.model_name = model_name
        print(f"Using device: {self.device}")
        print(f"Model: {model_name} ({num_channels}-channel)")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        # Torch 2.6 defaults to weights_only=True; allow full checkpoint for trusted source
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older torch without weights_only arg
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize IMU encoder based on model name
        if model_name == 'MW2StackRNNPooling18Ch':
            self.imu_encoder = MW2StackRNNPooling18Ch(size_embeddings=512)
        elif model_name == 'MW2StackRNNPooling9Ch':
            self.imu_encoder = MW2StackRNNPooling9Ch(size_embeddings=512)
        elif model_name == 'MW2StackRNNPooling21Ch':
            self.imu_encoder = MW2StackRNNPooling21Ch(size_embeddings=512)
        elif model_name == 'MW2StackRNNPooling':
            self.imu_encoder = MW2StackRNNPooling(size_embeddings=512)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Extract IMU encoder weights from checkpoint
        state_dict = checkpoint['state_dict']
        imu_state_dict = {
            k.replace('imu_encoder.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('imu_encoder.')
        }
        
        # Load weights (strict=False for flexibility)
        missing_keys, unexpected_keys = self.imu_encoder.load_state_dict(imu_state_dict, strict=False)
        self.imu_encoder.to(self.device)
        self.imu_encoder.eval()
        
        print(f"✓ IMU encoder loaded successfully")
        print(f"  Output dimension: 512 (CLIP embedding space)")
        if missing_keys:
            print(f"  Note: {len(missing_keys)} parameters not loaded (newly initialized): {missing_keys[:3]}...")

    def preprocess_imu(self, imu_data, sampling_rate=200, target_duration=5.0):
        """
        Preprocess IMU data to the format expected by the model
        
        Args:
            imu_data: numpy array of shape (N, 6/9) or (6/9, N)
                     6 channels: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                     9 channels: [accel_xyz, gyro_xyz, mag_xyz]
            sampling_rate: Hz (default 200)
            target_duration: seconds (default 5.0 for 2.5s * 2)
            
        Returns:
            torch.Tensor of shape (1, num_channels, target_samples)
        """
        # Convert to numpy if needed
        if isinstance(imu_data, torch.Tensor):
            imu_data = imu_data.cpu().numpy()
        
        # Handle different input shapes
        if len(imu_data.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {imu_data.shape}")
        
        # If shape is (N, channels), transpose to (channels, N)
        if imu_data.shape[0] > imu_data.shape[1]:
            imu_data = imu_data.T
        
        # Handle channel mismatch and rotation features for engineered modes
        if self.num_channels == 18 or self.num_channels == 21:
            # Both 18ch and 21ch modes start from 9 raw IMU channels
            if imu_data.shape[0] < 9:
                raise ValueError(f"{self.num_channels}-channel mode requires at least 9 IMU channels, got {imu_data.shape[0]}")
            elif imu_data.shape[0] > 9:
                print(f"Warning: Input has {imu_data.shape[0]} channels, using first 9 for {self.num_channels}-channel mode")
                imu_data = imu_data[:9, :]
        elif imu_data.shape[0] != self.num_channels:
            if imu_data.shape[0] > self.num_channels:
                print(f"Warning: Input has {imu_data.shape[0]} channels, using first {self.num_channels}")
                imu_data = imu_data[:self.num_channels, :]
            else:
                raise ValueError(f"Model expects {self.num_channels} channels, got {imu_data.shape[0]}")
        elif imu_data.shape[0] < 6:
            raise ValueError(f"Expected at least 6 channels, got {imu_data.shape[0]}")
        
        # Pad or trim to target length
        target_samples = int(target_duration * sampling_rate)
        current_samples = imu_data.shape[1]
        
        if current_samples < target_samples:
            # Pad with zeros
            n_channels = imu_data.shape[0]
            padding = np.zeros((n_channels, target_samples - current_samples))
            imu_data = np.concatenate([imu_data, padding], axis=1)
            print(f"Padded from {current_samples} to {target_samples} samples")
        elif current_samples > target_samples:
            # Trim
            imu_data = imu_data[:, :target_samples]
            print(f"Trimmed from {current_samples} to {target_samples} samples")
        
        # Compute engineered features
        if self.num_channels == 18:
            if imu_data.shape[0] != 9:
                raise ValueError(f"18-channel mode requires 9-channel data, got {imu_data.shape[0]}")
            try:
                engineered_features = compute_rotation_features(imu_data, sample_rate=sampling_rate)
                imu_data = np.concatenate([imu_data, engineered_features], axis=0)  # (18, N)
            except Exception as e:
                print(f"Warning: Engineered feature computation failed: {e}, using zeros")
                engineered_features = np.zeros((9, imu_data.shape[1]))
                imu_data = np.concatenate([imu_data, engineered_features], axis=0)
        elif self.num_channels == 21:
            if imu_data.shape[0] != 9:
                raise ValueError(f"21-channel mode requires 9-channel IMU input before feature computation, got {imu_data.shape[0]}")
            try:
                # Rotation features (9, N)
                rotation_features = compute_rotation_features(imu_data, sample_rate=sampling_rate)
                # World-frame linear acceleration (3, N)
                world_accel = compute_world_frame_acceleration(imu_data, rotation_features, sample_rate=sampling_rate)
                # Concatenate: 9 IMU + 9 rotation + 3 world accel = 21
                imu_data = np.concatenate([imu_data, rotation_features, world_accel], axis=0)
            except Exception as e:
                print(f"Warning: 21ch feature computation failed: {e}, using zeros")
                rotation_features = np.zeros((9, imu_data.shape[1]))
                world_accel = np.zeros((3, imu_data.shape[1]))
                imu_data = np.concatenate([imu_data, rotation_features, world_accel], axis=0)
        
        # Convert to tensor and add batch dimension
        imu_tensor = torch.FloatTensor(imu_data).unsqueeze(0)  # (1, num_channels, N)
        
        return imu_tensor

    @torch.no_grad()
    def encode(self, imu_data, sampling_rate=200):
        """
        Encode IMU data to CLIP embedding
        
        Args:
            imu_data: numpy array or torch tensor
                     Shape: (N, 6), (6, N), or (N, 9)
            sampling_rate: sampling rate in Hz
            
        Returns:
            numpy array of shape (512,) - CLIP embedding
        """
        # Preprocess
        imu_tensor = self.preprocess_imu(imu_data, sampling_rate)
        imu_tensor = imu_tensor.to(self.device)
        
        # Encode
        embedding = self.imu_encoder(imu_tensor)  # (1, 512)
        
        # Convert to numpy and remove batch dimension
        embedding = embedding.cpu().numpy()[0]  # (512,)
        
        return embedding
    
    @torch.no_grad()
    def encode_batch(self, imu_batch, sampling_rate=200):
        """
        Encode a batch of IMU sequences
        
        Args:
            imu_batch: list of numpy arrays or single array of shape (B, 6, N)
            sampling_rate: sampling rate in Hz
            
        Returns:
            numpy array of shape (B, 512)
        """
        if isinstance(imu_batch, list):
            # Process each individually and stack
            embeddings = [self.encode(imu, sampling_rate) for imu in imu_batch]
            return np.stack(embeddings)
        else:
            # Assume it's already batched (B, 6, N)
            imu_tensor = torch.FloatTensor(imu_batch).to(self.device)
            embeddings = self.imu_encoder(imu_tensor)
            return embeddings.cpu().numpy()
    
    def compute_similarity(self, imu_data1, imu_data2):
        """
        Compute cosine similarity between two IMU sequences
        
        Returns:
            float: similarity score [-1, 1]
        """
        emb1 = self.encode(imu_data1)
        emb2 = self.encode(imu_data2)
        
        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        return similarity


def load_imu_from_npy(npy_path):
    """Load IMU data from .npy file"""
    data = np.load(npy_path)
    print(f"Loaded IMU data: {data.shape}")
    return data


def load_imu_from_csv(csv_path):
    """Load IMU data from CSV file"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Try to extract IMU columns
    imu_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    if all(col in df.columns for col in imu_columns):
        data = df[imu_columns].values.T  # (6, N)
    else:
        # Assume all numeric columns are IMU data
        data = df.select_dtypes(include=[np.number]).values.T
    
    print(f"Loaded IMU data from CSV: {data.shape}")
    return data


if __name__ == "__main__":
    parser = ArgumentParser(description="IMU2CLIP Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./saved/i2c/i2c_s_i_t_t_se_mw2_w_2.5_master-epoch=01-val_loss=5.87.ckpt",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--imu_file",
        type=str,
        help="Path to IMU file (.npy or .csv)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu"
    )
    args = parser.parse_args()
    
    # Initialize model
    print("=" * 60)
    print("IMU2CLIP Inference")
    print("=" * 60)
    model = IMU2CLIPInference(args.checkpoint, device=args.device)
    
    # Test with sample data if no file provided
    if args.imu_file is None:
        print("\n" + "=" * 60)
        print("Demo: Testing with random IMU data")
        print("=" * 60)
        
        # Create random IMU data (6 channels, 1000 samples = 5 seconds at 200 Hz)
        sample_imu = np.random.randn(6, 1000)
        print(f"Sample IMU shape: {sample_imu.shape}")
        
        # Encode
        embedding = model.encode(sample_imu)
        print(f"\n✓ Encoded to CLIP embedding")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {np.linalg.norm(embedding):.4f}")
        print(f"  First 10 values: {embedding[:10]}")
        
        # Test similarity with another random sample
        sample_imu2 = np.random.randn(6, 1000)
        similarity = model.compute_similarity(sample_imu, sample_imu2)
        print(f"\n✓ Similarity between two random IMU sequences: {similarity:.4f}")
        
    else:
        # Load and encode actual IMU file
        print("\n" + "=" * 60)
        print(f"Processing: {args.imu_file}")
        print("=" * 60)
        
        if args.imu_file.endswith('.npy'):
            imu_data = load_imu_from_npy(args.imu_file)
        elif args.imu_file.endswith('.csv'):
            imu_data = load_imu_from_csv(args.imu_file)
        else:
            raise ValueError("Unsupported file format. Use .npy or .csv")
        
        # Encode
        embedding = model.encode(imu_data)
        print(f"\n✓ Encoded to CLIP embedding")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {np.linalg.norm(embedding):.4f}")
        
        # Save embedding
        output_path = args.imu_file.replace('.npy', '_embedding.npy').replace('.csv', '_embedding.npy')
        np.save(output_path, embedding)
        print(f"\n✓ Saved embedding to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

