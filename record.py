import os
import sys
import time
import argparse
from pathlib import Path

# sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))

from collection.collect_one_traj import DataCollector
import collection.hdf5_to_csv as hdf5_to_csv
import collection.hdf5_2_mp4 as hdf5_2_mp4

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Record IMU and video data for a specific task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python record.py --folder_name test_data --task_name pick_and_place --webcam_port 2'
    )
    
    parser.add_argument(
        '--folder_name',
        type=str,
        required=True,
        help='Folder name to store the data (e.g., test_data, nov2_set)'
    )
    
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        help='Task name for the recording (e.g., pick_and_place, wipe_table)'
    )
    
    parser.add_argument(
        '--webcam_port',
        type=int,
        default=0,
        help='Webcam port number (default: 0). Use check_camera_ports.py to find available ports.'
    )
    
    args = parser.parse_args()
    
    # Configuration from arguments
    folder_name = args.folder_name
    task_name = args.task_name
    path = Path("data") / folder_name / task_name
    
    # Create output directory
    output_dir = path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_hdf5 = output_dir / f"{task_name}_{timestamp}.hdf5"
    
    print(f"Starting data collection for task: {task_name}")
    print(f"Webcam port: {args.webcam_port}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_hdf5}")
    
    try:
        # Create data collector
        collector = DataCollector(
            serial_port="/dev/ttyACM0",
            baud_rate=115200,
            webcam_port=args.webcam_port,
            imu_freq=200,  # 200 Hz
            camera_freq=30,  # 30 FPS
            output_hdf5=str(output_hdf5),
            duration=7
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