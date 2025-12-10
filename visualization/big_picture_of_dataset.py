#!/usr/bin/env python3
"""
Create a grid visualization of dataset showing many video samples.
Grid: 10 rows x 20 columns = 200 video samples
No text/labels, just pure video frames.
"""

import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import random

# --- CONFIGURATION ---
DATA_ROOT = Path('/home/chuye/Documents/MCI-project/data/all_data')
OUTPUT_PATH = Path('/home/chuye/Documents/MCI-project/data/dataset_big_picture.gif')

GRID_ROWS = 10
GRID_COLS = 20
TOTAL_SAMPLES = GRID_ROWS * GRID_COLS  # 200

FPS = 15
FRAME_STRIDE = 2  # Skip frames to reduce file size
THUMBNAIL_SIZE = (80, 60)  # Size of each video thumbnail
RANDOM_SEED = 42
# ---------------------

def find_all_hdf5_files(root_dir: Path):
    """Recursively find all HDF5 files in directory."""
    return list(root_dir.rglob('*.hdf5'))

def load_video_frames(hdf5_path: Path):
    """Load and convert video frames from HDF5."""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'video' not in f:
                return None
            frames = f['video']['frames'][()]
            # Convert BGR to RGB
            frames = frames[..., ::-1]
            return frames
    except Exception as e:
        print(f"⚠️  Error loading {hdf5_path.name}: {e}")
        return None

def resize_frame(frame, size):
    """Resize frame to thumbnail size."""
    img = Image.fromarray(frame)
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)

def create_grid_gif():
    """Create a grid GIF from many video samples."""
    
    print("="*70)
    print("Creating Dataset Big Picture GIF")
    print(f"Grid: {GRID_ROWS} rows × {GRID_COLS} columns = {TOTAL_SAMPLES} samples")
    print("="*70)
    print()
    
    # 1. Find all HDF5 files
    print("🔍 Finding HDF5 files...")
    all_files = find_all_hdf5_files(DATA_ROOT)
    print(f"   Found {len(all_files)} files")
    
    if len(all_files) == 0:
        print("❌ No HDF5 files found!")
        return
    
    # 2. Randomly sample files
    random.seed(RANDOM_SEED)
    num_to_sample = min(TOTAL_SAMPLES, len(all_files))
    sampled_files = random.sample(all_files, num_to_sample)
    
    print(f"📝 Sampling {num_to_sample} files...")
    print()
    
    # 3. Load video data from sampled files
    print("📹 Loading video frames...")
    video_data = []
    max_frames = 0
    
    for i, file_path in enumerate(sampled_files):
        if (i + 1) % 20 == 0:
            print(f"   Loaded {i+1}/{num_to_sample} videos...")
        
        frames = load_video_frames(file_path)
        if frames is not None and len(frames) > 0:
            video_data.append(frames)
            max_frames = max(max_frames, len(frames))
        else:
            # Use black placeholder if video fails to load
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            video_data.append(np.array([black_frame]))
    
    print(f"   ✅ Loaded {len(video_data)} videos")
    print(f"   Max frames: {max_frames}")
    print()
    
    # 4. Generate GIF frames
    print(f"🎬 Generating GIF frames (every {FRAME_STRIDE} frames)...")
    
    gif_frames = []
    frame_indices = range(0, max_frames, FRAME_STRIDE)
    
    grid_height = GRID_ROWS * THUMBNAIL_SIZE[1]
    grid_width = GRID_COLS * THUMBNAIL_SIZE[0]
    
    for frame_idx in frame_indices:
        if len(gif_frames) % 20 == 0 and len(gif_frames) > 0:
            print(f"   Generated {len(gif_frames)} frames...")
        
        # Create empty grid for this frame
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Fill grid with video frames
        for vid_idx, video in enumerate(video_data):
            row = vid_idx // GRID_COLS
            col = vid_idx % GRID_COLS
            
            # Get current frame (or last frame if video ended)
            curr_frame_idx = min(frame_idx, len(video) - 1)
            frame = video[curr_frame_idx]
            
            # Resize to thumbnail
            thumb = resize_frame(frame, THUMBNAIL_SIZE)
            
            # Place in grid
            y_start = row * THUMBNAIL_SIZE[1]
            y_end = y_start + THUMBNAIL_SIZE[1]
            x_start = col * THUMBNAIL_SIZE[0]
            x_end = x_start + THUMBNAIL_SIZE[0]
            
            grid[y_start:y_end, x_start:x_end] = thumb
        
        # Convert to PIL Image
        gif_frames.append(Image.fromarray(grid))
    
    print(f"   ✅ Generated {len(gif_frames)} frames")
    print()
    
    # 5. Save GIF
    print("💾 Saving GIF...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if len(gif_frames) > 0:
        gif_frames[0].save(
            OUTPUT_PATH,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / FPS),
            loop=0,
            optimize=False
        )
        
        file_size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
        
        print("="*70)
        print("✅ GIF created successfully!")
        print(f"   Path: {OUTPUT_PATH}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Dimensions: {grid_width}×{grid_height}")
        print(f"   Frames: {len(gif_frames)}")
        print(f"   FPS: {FPS}")
        print("="*70)
    else:
        print("❌ No frames generated!")

if __name__ == "__main__":
    create_grid_gif()
