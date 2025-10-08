import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))

from collect_one_traj import DataCollector
import hdf5_to_csv
import hdf5_2_mp4

def main():
    # Configuration
    folder_name = "test_data"
    task_name = "wipe_table"
    path = Path("data") / folder_name / task_name
    
    # Create output directory
    output_dir = path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_hdf5 = output_dir / f"{task_name}_{timestamp}.hdf5"
    
    print(f"Starting data collection for task: {task_name}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_hdf5}")
    
    try:
        # Create data collector
        collector = DataCollector(
            serial_port="/dev/ttyACM0",
            baud_rate=115200,
            webcam_port=0,
            imu_freq=200,  # 200 Hz
            camera_freq=30,  # 30 FPS
            output_hdf5=str(output_hdf5)
        )
        
        # Start recording for 15 seconds
        print("\n" + "="*50)
        print("STARTING RECORDING...")
        print("="*50)
        
        success = collector.start_recording(duration=15)
        
        if success:
            print("\n" + "="*50)
            print("RECORDING COMPLETED SUCCESSFULLY!")
            print("="*50)
            
            # Convert HDF5 to CSV
            print("\nConverting HDF5 to CSV...")
            csv_path = hdf5_to_csv.hdf5_to_csv(str(output_hdf5))
            
            # Convert HDF5 to MP4
            print("\nConverting HDF5 to MP4...")
            mp4_path = hdf5_2_mp4.hdf5_to_mp4(str(output_hdf5))
            
            print(f"\n🎉 Data collection and conversion completed!")
            print(f"📁 HDF5 file: {output_hdf5}")
            print(f"📊 CSV file: {csv_path}")
            print(f"🎬 MP4 file: {mp4_path}")
            
        else:
            print("❌ Recording failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())