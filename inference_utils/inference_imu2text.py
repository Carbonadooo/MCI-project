#!/usr/bin/env python3
"""
IMU2Text Inference: Compare IMU motions with text descriptions
Uses both IMU encoder and CLIP text encoder
"""

import torch
import numpy as np
import clip
from inference_imu2clip import IMU2CLIPInference, load_imu_from_npy
from argparse import ArgumentParser


class IMU2TextMatcher:
    """
    Match IMU sequences to text descriptions using CLIP embedding space
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize IMU2Text matcher
        
        Args:
            checkpoint_path: Path to trained IMU2CLIP checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load IMU encoder
        print("Loading IMU encoder...")
        self.imu_model = IMU2CLIPInference(checkpoint_path, device=device)
        
        # Load CLIP text encoder
        print("Loading CLIP text encoder...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        print("✓ Both encoders loaded successfully\n")
    
    @torch.no_grad()
    def encode_text(self, text_descriptions):
        """
        Encode text descriptions to CLIP embeddings
        
        Args:
            text_descriptions: string or list of strings
            
        Returns:
            numpy array of shape (N, 512) or (512,)
        """
        if isinstance(text_descriptions, str):
            text_descriptions = [text_descriptions]
            single = True
        else:
            single = False
        
        # Tokenize
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        
        # Encode
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features.cpu().numpy()
        
        # Normalize
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        
        return text_features[0] if single else text_features
    
    def match_imu_to_texts(self, imu_data, text_descriptions):
        """
        Match an IMU sequence to a list of text descriptions
        
        Args:
            imu_data: numpy array (6, N) or (N, 6)
            text_descriptions: list of strings
            
        Returns:
            dict with similarities and best match
        """
        # Encode IMU
        imu_embedding = self.imu_model.encode(imu_data)
        imu_embedding = imu_embedding / np.linalg.norm(imu_embedding)  # Normalize
        
        # Encode texts
        text_embeddings = self.encode_text(text_descriptions)
        
        # Compute similarities
        similarities = np.dot(text_embeddings, imu_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        
        results = {
            'similarities': {
                text: float(sim) 
                for text, sim in zip(text_descriptions, similarities)
            },
            'best_match': text_descriptions[best_idx],
            'best_score': float(similarities[best_idx])
        }
        
        return results
    
    def find_similar_imus(self, query_imu, candidate_imus):
        """
        Find most similar IMU sequences to a query IMU
        
        Args:
            query_imu: numpy array (6, N)
            candidate_imus: list of numpy arrays
            
        Returns:
            list of (index, similarity) sorted by similarity
        """
        # Encode query
        query_embedding = self.imu_model.encode(query_imu)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Encode candidates
        similarities = []
        for i, candidate in enumerate(candidate_imus):
            cand_embedding = self.imu_model.encode(candidate)
            cand_embedding = cand_embedding / np.linalg.norm(cand_embedding)
            
            sim = np.dot(query_embedding, cand_embedding)
            similarities.append((i, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities


def demo_text_retrieval(matcher):
    """Demo: Retrieve text descriptions for IMU motion"""
    print("=" * 70)
    print("Demo 1: IMU-to-Text Retrieval")
    print("=" * 70)
    
    # Sample text descriptions
    text_descriptions = [
        "person walking forward",
        "person running fast",
        "person sitting down",
        "person standing still",
        "person jumping up and down",
        "person turning left",
        "person opening a door",
        "person climbing stairs"
    ]
    
    # Create sample IMU (simulate walking motion)
    # In practice, load from your actual data
    sample_imu = np.random.randn(6, 1000)
    
    # Match
    results = matcher.match_imu_to_texts(sample_imu, text_descriptions)
    
    print(f"Best match: '{results['best_match']}'")
    print(f"Confidence: {results['best_score']:.4f}\n")
    
    print("All similarities:")
    for text, sim in sorted(results['similarities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {sim:.4f} - {text}")
    print()

def load_imu_from_hdf5(hdf5_path):
    """
    Load IMU data from HDF5 file
    
    Args:
        hdf5_path: Path to .hdf5 file containing IMU data
        
    Returns:
        dict with:
            - 'imu_data': numpy array (6, N) - 6 channels of IMU data
            - 'timestamps': numpy array (N,) - timestamps
            - 'metadata': dict - file metadata
    """
    import h5py
    
    print(f"Loading HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as hf:
        # Load IMU data
        imu_timestamps = hf['imu/timestamps'][:]
        imu_data = hf['imu/data'][:]
        
        # Get metadata if available
        metadata = dict(hf.attrs) if hasattr(hf, 'attrs') else {}
        
        print(f"  IMU data shape: {imu_data.shape}")
        print(f"  Timestamps shape: {imu_timestamps.shape}")
        print(f"  Duration: {(imu_timestamps[-1] - imu_timestamps[0]):.2f} seconds")
    
    # Convert to (6, N) format if needed
    if len(imu_data.shape) == 2:
        if imu_data.shape[0] > imu_data.shape[1]:
            # Shape is (N, channels), transpose to (channels, N)
            imu_data = imu_data.T
        
        # Extract first 6 channels if more are present
        if imu_data.shape[0] > 6:
            print(f"  Using first 6 channels (found {imu_data.shape[0]})")
            imu_data = imu_data[:6, :]
        elif imu_data.shape[0] < 6:
            raise ValueError(f"Expected at least 6 IMU channels, got {imu_data.shape[0]}")
    else:
        raise ValueError(f"Expected 2D IMU data, got shape {imu_data.shape}")
    
    return {
        'imu_data': imu_data,
        'timestamps': imu_timestamps,
        'metadata': metadata
    }


def demo_hdf5_inference(matcher, hdf5_path):
    """
    Demo: Load HDF5 file and run inference
    
    Args:
        matcher: IMU2TextMatcher instance
        hdf5_path: Path to HDF5 file
    """
    print("=" * 70)
    print("Demo: HDF5 File Inference")
    print("=" * 70)
    
    # Load HDF5 file
    data = load_imu_from_hdf5(hdf5_path)
    imu_data = data['imu_data']
    
    print(f"\n✓ Loaded IMU data: {imu_data.shape}")
    
    # Define some action descriptions to match against
    action_descriptions = [
        "person clapping hands",
        "person walking",
        "person running",
        "person sitting down",
        "person standing up",
        "person waving hand",
        "person jumping",
        "person turning around"
    ]
    
    # Match IMU to text descriptions
    print("\nMatching IMU motion to text descriptions...")
    results = matcher.match_imu_to_texts(imu_data, action_descriptions)
    
    print(f"\n✓ Best match: '{results['best_match']}'")
    print(f"  Confidence: {results['best_score']:.4f}\n")
    
    print("All similarities:")
    for text, sim in sorted(results['similarities'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(sim * 50) if sim > 0 else ''
        print(f"  {sim:6.4f} {bar:50s} {text}")
    
    print()

def demo_imu_retrieval(matcher, imu_dir):
    """Demo: Find similar IMU sequences"""
    print("=" * 70)
    print("Demo 2: IMU-to-IMU Retrieval")
    print("=" * 70)
    
    import glob
    
    # Load a few IMU samples
    imu_files = sorted(glob.glob(f"{imu_dir}/*.npy"))[:5]
    imu_files = [f for f in imu_files if not f.endswith('_timestamps.npy')]
    
    if len(imu_files) < 2:
        print("Not enough IMU files for demo")
        return
    
    print(f"Loading {len(imu_files)} IMU files...")
    imus = [load_imu_from_npy(f) for f in imu_files]
    
    # Use first as query
    query_imu = imus[0]
    candidate_imus = imus[1:]
    
    print(f"\nQuery: {imu_files[0].split('/')[-1]}")
    print("\nFinding similar IMU sequences...")
    
    similarities = matcher.find_similar_imus(query_imu, candidate_imus)
    
    print("\nTop 3 most similar:")
    for i, (idx, sim) in enumerate(similarities[:3]):
        filename = imu_files[idx + 1].split('/')[-1]
        print(f"  {i+1}. {sim:.4f} - {filename}")
    print()


def demo_zero_shot_classification(matcher):
    """Demo: Zero-shot action classification"""
    print("=" * 70)
    print("Demo 3: Zero-Shot Action Classification")
    print("=" * 70)
    
    # Define action classes
    action_classes = [
        "walking",
        "running",
        "sitting",
        "standing",
        "climbing",
        "descending stairs"
    ]
    
    # Create sample IMU
    sample_imu = np.random.randn(6, 1000)
    
    # Classify
    results = matcher.match_imu_to_texts(sample_imu, action_classes)
    
    print(f"Predicted action: {results['best_match']}")
    print(f"Confidence: {results['best_score']:.4f}\n")
    
    print("Class probabilities (softmax):")
    scores = np.array(list(results['similarities'].values()))
    probs = np.exp(scores) / np.sum(np.exp(scores))
    
    for action, prob in zip(action_classes, probs):
        bar = '█' * int(prob * 50)
        print(f"  {action:20s} {bar} {prob:.3f}")
    print()


if __name__ == "__main__":
    parser = ArgumentParser(description="IMU2Text Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./saved/i2c/i2c_s_i_t_t_se_mw2_w_2.5_master-epoch=01-val_loss=5.87.ckpt",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--imu_dir",
        type=str,
        default="./checkpoint/clips/processed_imu",
        help="Directory with IMU files for demo"
    )
    parser.add_argument(
        "--hdf5_file",
        type=str,
        default=None,
        help="Path to HDF5 file with IMU data (e.g., /path/to/clap_once.hdf5)"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=['all', 'text', 'imu', 'classify', 'hdf5'],
        default='all',
        help="Which demo to run"
    )
    args = parser.parse_args()
    
    # Initialize matcher
    print("=" * 70)
    print("IMU2Text Matcher")
    print("=" * 70)
    matcher = IMU2TextMatcher(args.checkpoint, device=args.device)
    
    # Run HDF5 demo if file is provided
    if args.hdf5_file:
        demo_hdf5_inference(matcher, args.hdf5_file)
    elif args.demo == 'hdf5':
        print("Error: --hdf5_file must be provided for hdf5 demo")
        print("Example: --demo hdf5 --hdf5_file /path/to/your/data.hdf5")
    else:
        # Run other demos
        if args.demo in ['all', 'text']:
            demo_text_retrieval(matcher)
        
        if args.demo in ['all', 'classify']:
            demo_zero_shot_classification(matcher)
        
        if args.demo in ['all', 'imu']:
            demo_imu_retrieval(matcher, args.imu_dir)
    
    print("=" * 70)
    print("Done!")
    print("=" * 70)

