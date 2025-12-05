#!/usr/bin/env python3
"""
Test script to check if HDF5 data loading is working correctly
"""
import sys
from pathlib import Path
import glob
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from inference_hdf5 import load_hdf5_imu

def test_hdf5_loading(data_dir):
    """Test loading HDF5 files from a directory"""
    print(f"Testing HDF5 loading from: {data_dir}")
    print("=" * 70)
    
    # Find all HDF5 files
    hdf5_files = sorted(glob.glob(str(Path(data_dir) / "**/*.hdf5"), recursive=True))
    print(f"Found {len(hdf5_files)} HDF5 files")
    print()
    
    nan_files = []
    inf_files = []
    good_files = []
    
    for i, hdf5_file in enumerate(hdf5_files):
        try:
            data = load_hdf5_imu(hdf5_file, verbose=False, resample_to_200hz=True)
            imu_data = data['imu_data']
            
            has_nan = np.isnan(imu_data).any()
            has_inf = np.isinf(imu_data).any()
            
            if has_nan:
                nan_count = np.isnan(imu_data).sum()
                nan_files.append((hdf5_file, nan_count))
                print(f"❌ NaN in file {i+1}/{len(hdf5_files)}: {Path(hdf5_file).name}")
                print(f"   NaN count: {nan_count} / {imu_data.size}")
            
            if has_inf:
                inf_count = np.isinf(imu_data).sum()
                inf_files.append((hdf5_file, inf_count))
                print(f"❌ Inf in file {i+1}/{len(hdf5_files)}: {Path(hdf5_file).name}")
                print(f"   Inf count: {inf_count} / {imu_data.size}")
            
            if not has_nan and not has_inf:
                good_files.append(hdf5_file)
                if i < 5:  # Show stats for first 5 files
                    print(f"✓ File {i+1}: {Path(hdf5_file).name}")
                    print(f"    Shape: {imu_data.shape}")
                    print(f"    Mean: {imu_data.mean():.4f}, Std: {imu_data.std():.4f}")
                    print(f"    Min: {imu_data.min():.4f}, Max: {imu_data.max():.4f}")
        
        except Exception as e:
            print(f"❌ ERROR loading file {i+1}: {Path(hdf5_file).name}")
            print(f"   Error: {e}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(hdf5_files)}")
    print(f"Good files:  {len(good_files)} ({100*len(good_files)/len(hdf5_files):.1f}%)")
    print(f"Files with NaN: {len(nan_files)}")
    print(f"Files with Inf: {len(inf_files)}")
    
    if nan_files:
        print("\nFiles with NaN:")
        for file, count in nan_files[:10]:  # Show first 10
            print(f"  - {Path(file).name}: {count} NaN values")
        if len(nan_files) > 10:
            print(f"  ... and {len(nan_files) - 10} more")
    
    if inf_files:
        print("\nFiles with Inf:")
        for file, count in inf_files[:10]:
            print(f"  - {Path(file).name}: {count} Inf values")
        if len(inf_files) > 10:
            print(f"  ... and {len(inf_files) - 10} more")
    
    print()
    
    if nan_files or inf_files:
        print("⚠️  WARNING: Found problematic files! These need to be cleaned.")
        print("   Consider:")
        print("   1. Re-recording the problematic samples")
        print("   2. Removing them from the dataset")
        print("   3. Using NaN replacement (already implemented)")
    else:
        print("✓ All files are clean!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test HDF5 data loading")
    parser.add_argument("data_dir", type=str, help="Data directory to test")
    args = parser.parse_args()
    
    test_hdf5_loading(args.data_dir)

