#!/usr/bin/env python3
"""
Split dataset into training and validation sets.

Structure:
  all_data/task_name/*.hdf5, *.csv, *.mp4
    → train_set/task_name/*.hdf5, *.csv, *.mp4
    → validate_set/task_name/*.hdf5, *.csv, *.mp4

For each task: 8 trajectories → validation, rest → training
"""

import shutil
from pathlib import Path
import random
from collections import defaultdict

# --- CONFIGURATION ---
SOURCE_DIR = '/home/chuye/Documents/MCI-project/data/all_data'
TRAIN_DIR = '/home/chuye/Documents/MCI-project/data/train_set'
VAL_DIR = '/home/chuye/Documents/MCI-project/data/validate_set'
VAL_COUNT = 8  # Number of trajectories per task for validation
RANDOM_SEED = 42  # For reproducible splits
# ---------------------

def get_trajectory_groups(task_dir: Path):
    """
    Group files by trajectory (same base name, different extensions).
    
    Returns: dict mapping base_name -> list of file paths
    
    Example:
      'clap_once_20251108_191951' -> [
          'clap_once_20251108_191951.hdf5',
          'clap_once_20251108_191951.mp4',
          'clap_once_20251108_191951_imu.csv'
      ]
    """
    trajectories = defaultdict(list)
    
    # Find all files (excluding description.txt)
    for file_path in task_dir.iterdir():
        if file_path.is_file() and file_path.name != 'description.txt':
            # Extract base name (remove _imu.csv or just extension)
            name = file_path.stem
            if name.endswith('_imu'):
                base_name = name[:-4]  # Remove '_imu'
            else:
                base_name = name
            
            trajectories[base_name].append(file_path)
    
    return trajectories

def split_and_copy(source_root: Path, train_root: Path, val_root: Path, val_count: int, seed: int):
    """
    Split dataset into train and validation sets.
    """
    source_path = Path(source_root)
    train_path = Path(train_root)
    val_path = Path(val_root)
    
    if not source_path.exists():
        print(f"❌ Error: Source directory not found: {source_path}")
        return
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all task directories (subdirectories in source)
    task_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not task_dirs:
        print(f"⚠️  No task directories found in {source_path}")
        return
    
    print("="*70)
    print(f"Dataset Splitting")
    print(f"Source: {source_path}")
    print(f"Train output: {train_path}")
    print(f"Validation output: {val_path}")
    print(f"Validation trajectories per task: {val_count}")
    print(f"Random seed: {seed}")
    print("="*70)
    print()
    
    total_train = 0
    total_val = 0
    
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        print(f"📁 Processing task: {task_name}")
        
        # Get trajectory groups
        trajectories = get_trajectory_groups(task_dir)
        
        if not trajectories:
            print(f"   ⚠️  No trajectories found, skipping...")
            continue
        
        # Get list of trajectory base names and shuffle
        traj_names = list(trajectories.keys())
        random.shuffle(traj_names)
        
        # Split into validation and training
        val_names = traj_names[:val_count]
        train_names = traj_names[val_count:]
        
        print(f"   Total trajectories: {len(traj_names)}")
        print(f"   → Validation: {len(val_names)}")
        print(f"   → Training: {len(train_names)}")
        
        # Create output directories
        train_task_dir = train_path / task_name
        val_task_dir = val_path / task_name
        train_task_dir.mkdir(parents=True, exist_ok=True)
        val_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy validation files
        for traj_name in val_names:
            for file_path in trajectories[traj_name]:
                dest_path = val_task_dir / file_path.name
                shutil.copy2(file_path, dest_path)
        
        # Copy training files
        for traj_name in train_names:
            for file_path in trajectories[traj_name]:
                dest_path = train_task_dir / file_path.name
                shutil.copy2(file_path, dest_path)
        
        # Copy description.txt if exists
        # desc_file = task_dir / 'description.txt'
        # if desc_file.exists():
        #     shutil.copy2(desc_file, train_task_dir / 'description.txt')
        #     shutil.copy2(desc_file, val_task_dir / 'description.txt')
        
        total_train += len(train_names)
        total_val += len(val_names)
        print(f"   ✅ Done")
        print()
    
    print("="*70)
    print("Split Summary")
    print("="*70)
    print(f"Total tasks: {len(task_dirs)}")
    print(f"Total training trajectories: {total_train}")
    print(f"Total validation trajectories: {total_val}")
    print(f"Total trajectories: {total_train + total_val}")
    print()
    print("✅ Dataset split completed!")
    print("="*70)

if __name__ == "__main__":
    split_and_copy(SOURCE_DIR, TRAIN_DIR, VAL_DIR, VAL_COUNT, RANDOM_SEED)

