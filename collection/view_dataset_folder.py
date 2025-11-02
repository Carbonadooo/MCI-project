import os
from pathlib import Path
from datetime import datetime
import h5py
DATASET_PATH = "/home/chuye/Documents/MCI-project/data/nov2_set"

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0

def analyze_dataset_folder(dataset_path):
    """Analyze dataset folder structure and generate description"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return None
    
    print(f"Analyzing dataset folder: {dataset_path}")
    
    # Dictionary to store task information
    tasks_info = {}
    
    # Scan all subfolders (task names)
    subfolders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print(f"⚠️  No task subfolders found in {dataset_path}")
        return None
    
    total_trajectories = 0
    total_files = 0
    total_size_mb = 0
    
    # Analyze each task folder
    for task_folder in sorted(subfolders):
        task_name = task_folder.name
        task_path = task_folder
        
        # Count files and trajectories
        files = list(task_path.glob("*"))
        hdf5_files = list(task_path.glob("*.hdf5"))
        csv_files = list(task_path.glob("*.csv"))
        mp4_files = list(task_path.glob("*.mp4"))
        
        # Count trajectories (unique recordings based on HDF5 files)
        num_trajectories = len(hdf5_files)
        num_files = len(files)
        
        # Calculate total size for this task
        task_size_mb = sum(get_file_size_mb(f) for f in files)
        
        # Get recording metadata from first HDF5 if available
        metadata = {}
        if hdf5_files:
            try:
                with h5py.File(hdf5_files[0], 'r') as hf:
                    metadata['imu_freq'] = hf['imu'].attrs.get('frequency', 'unknown')
                    metadata['camera_freq'] = hf['video'].attrs.get('fps', 'unknown')
                    metadata['imu_samples'] = len(hf['imu/timestamps'][:]) if 'imu/timestamps' in hf else 0
                    metadata['video_frames'] = len(hf['video/frames'][:]) if 'video/frames' in hf else 0
                    metadata['created_at'] = hf.attrs.get('created_at', 'unknown')
            except Exception as e:
                print(f"  ⚠️  Could not read metadata from {hdf5_files[0]}: {e}")
        
        # Get date range of recordings
        dates = []
        for hdf5_file in hdf5_files:
            try:
                # Extract timestamp from filename (format: task_name_YYYYMMDD_HHMMSS.hdf5)
                name = hdf5_file.stem
                if '_' in name:
                    parts = name.split('_')
                    if len(parts) >= 3:
                        date_str = f"{parts[-2]}_{parts[-1]}"
                        try:
                            date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                            dates.append(date)
                        except:
                            pass
            except:
                pass
        
        date_range = ""
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            if min_date.date() == max_date.date():
                date_range = min_date.strftime("%Y-%m-%d")
            else:
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        
        tasks_info[task_name] = {
            'num_trajectories': num_trajectories,
            'num_files': num_files,
            'hdf5_files': len(hdf5_files),
            'csv_files': len(csv_files),
            'mp4_files': len(mp4_files),
            'size_mb': task_size_mb,
            'date_range': date_range,
            'metadata': metadata
        }
        
        total_trajectories += num_trajectories
        total_files += num_files
        total_size_mb += task_size_mb
    
    # Generate description text
    description = []
    description.append("=" * 80)
    description.append("DATASET DESCRIPTION")
    description.append("=" * 80)
    description.append(f"\nDataset Path: {dataset_path}")
    description.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    description.append("\n" + "-" * 80)
    description.append("SUMMARY")
    description.append("-" * 80)
    description.append(f"Total Tasks: {len(tasks_info)}")
    description.append(f"Total Trajectories: {total_trajectories}")
    description.append(f"Total Files: {total_files}")
    description.append(f"Total Size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    description.append("\n" + "-" * 80)
    description.append("TASK DETAILS")
    description.append("-" * 80)
    
    # Sort tasks by name
    for task_name in sorted(tasks_info.keys()):
        info = tasks_info[task_name]
        description.append(f"\n📁 Task: {task_name}")
        description.append(f"   • Trajectories: {info['num_trajectories']}")
        description.append(f"   • Total Files: {info['num_files']} (HDF5: {info['hdf5_files']}, CSV: {info['csv_files']}, MP4: {info['mp4_files']})")
        description.append(f"   • Size: {info['size_mb']:.2f} MB")
        
        if info['date_range']:
            description.append(f"   • Date Range: {info['date_range']}")
        
        if info['metadata']:
            meta = info['metadata']
            if meta.get('imu_freq') and meta.get('camera_freq'):
                description.append(f"   • IMU Frequency: {meta['imu_freq']} Hz")
                description.append(f"   • Camera Frequency: {meta['camera_freq']} FPS")
            
            if meta.get('imu_samples') and meta.get('video_frames'):
                avg_samples = meta['imu_samples'] / info['num_trajectories'] if info['num_trajectories'] > 0 else 0
                avg_frames = meta['video_frames'] / info['num_trajectories'] if info['num_trajectories'] > 0 else 0
                description.append(f"   • Avg IMU Samples/Trajectory: {avg_samples:.0f}")
                description.append(f"   • Avg Video Frames/Trajectory: {avg_frames:.0f}")
    
    description.append("\n" + "=" * 80)
    description.append("END OF DESCRIPTION")
    description.append("=" * 80)
    
    # Write to file
    output_file = dataset_path / "description.txt"
    description_text = "\n".join(description)
    
    with open(output_file, 'w') as f:
        f.write(description_text)
    
    print(f"\n✅ Dataset analysis completed!")
    print(f"📄 Description saved to: {output_file}")
    print("\n" + description_text)
    
    return {
        'tasks_info': tasks_info,
        'total_trajectories': total_trajectories,
        'total_files': total_files,
        'total_size_mb': total_size_mb,
        'description': description_text
    }

if __name__ == "__main__":
    analyze_dataset_folder(DATASET_PATH)