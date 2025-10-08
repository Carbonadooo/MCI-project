import h5py
import pandas as pd
import numpy as np
import os
from pathlib import Path

def hdf5_to_csv(input_hdf5_path, output_csv_path=None):
    """Convert HDF5 IMU data to CSV format"""
    
    print(f"Converting HDF5 IMU data to CSV: {input_hdf5_path}")
    
    # Validate input file
    if not os.path.exists(input_hdf5_path):
        raise FileNotFoundError(f"Input file not found: {input_hdf5_path}")
    
    # Generate output path if not provided
    if output_csv_path is None:
        input_path = Path(input_hdf5_path)
        output_csv_path = input_path.parent / f"{input_path.stem}_imu.csv"
    
    try:
        # Open HDF5 file
        with h5py.File(input_hdf5_path, 'r') as hf:
            print("Reading IMU data from HDF5...")
            
            # Check if IMU data exists
            if 'imu/data' not in hf or 'imu/timestamps' not in hf:
                raise ValueError("No IMU data found in HDF5 file")
            
            # Load IMU data
            imu_timestamps = hf['imu/timestamps'][:]
            imu_data = hf['imu/data'][:]
            
            # Get metadata
            imu_attrs = dict(hf['imu'].attrs)
            
            print(f"Loaded {len(imu_timestamps)} IMU samples")
            print(f"IMU frequency: {imu_attrs.get('frequency', 'unknown')} Hz")
            
            # Create DataFrame
            columns = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
            df = pd.DataFrame(imu_data, columns=columns[1:])  # Skip timestamp column
            df.insert(0, 'timestamp', imu_timestamps)
            
            # Save to CSV
            df.to_csv(output_csv_path, index=False)
            
            print(f"✅ IMU data conversion completed!")
            print(f"Output file: {output_csv_path}")
            print(f"Total samples: {len(df)}")
            print(f"Duration: {(imu_timestamps[-1] - imu_timestamps[0]):.2f} seconds")
            
            return str(output_csv_path)
            
    except Exception as e:
        print(f"❌ Error converting HDF5 to CSV: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "/home/chuye/Documents/MCI-project/data/output_data.hdf5"
    
    try:
        output_path = hdf5_to_csv(input_file)
        print(f"\n🎉 Successfully converted to: {output_path}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
