import serial
import csv
import time
import threading
import cv2
import numpy as np
import h5py
import os

class DataCollector:
    def __init__(self, serial_port="/dev/ttyACM0", baud_rate=115200, 
                 webcam_port=0, imu_freq=200, camera_freq=30, output_hdf5="./data/output_data.hdf5"):
        # Configuration
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.webcam_port = webcam_port
        self.imu_frequency = imu_freq  # Hz
        self.camera_frequency = camera_freq  # FPS
        
        # File paths
        self.tmp_output_csv = "./tmp_imu_data.csv"
        self.tmp_output_mp4 = "./tmp_video_data.mp4"
        self.output_hdf5 = output_hdf5
        
        # Camera settings
        self.camera_width = 640
        self.camera_height = 480
        
        # Hardware interfaces (initialized once)
        self.ser = None
        self.cap = None
        self.video_writer = None
        
        # Data storage
        self.imu_data = []
        self.imu_timestamps = []
        self.video_frames = []
        self.video_timestamps = []
        
        # Threading
        self.recording = False
        self.imu_thread = None
        self.video_thread = None
        self.data_lock = threading.Lock()
        
        # Timing
        self.start_time = None
        self.imu_interval = 1.0 / self.imu_frequency  # seconds between samples
        self.camera_interval = 1.0 / self.camera_frequency  # seconds between frames

    def initialize_hardware(self):
        """Initialize serial port and camera once"""
        try:
            # Initialize serial port
            print(f"Initializing serial port: {self.serial_port}")
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)  
            print("Serial port initialized successfully")
            
            # Initialize camera
            print(f"Initializing camera: {self.webcam_port}")
            self.cap = cv2.VideoCapture(self.webcam_port)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_frequency)

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read from camera")
            print("Camera initialized successfully")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.tmp_output_mp4, fourcc, self.camera_frequency, 
                (self.camera_width, self.camera_height)
            )
            print("Video writer initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"Hardware initialization error: {e}")
            self.cleanup_hardware()
            return False
    
    def cleanup_hardware(self):
        """Clean up hardware resources"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")
        
        if self.cap:
            self.cap.release()
            print("Camera released")
        
        if self.video_writer:
            self.video_writer.release()
            print("Video writer released")
        
        cv2.destroyAllWindows()
    
    def record_imu_data(self):
        """Thread function to record IMU data at fixed frequency"""
        print(f"IMU recording started at {self.imu_frequency} Hz...")
        
        # Open CSV file for writing
        csv_file = open(self.tmp_output_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp","ax","ay","az","gx","gy","gz","mx","my","mz"])
        
        try:
            # Use precise timing for consistent sampling
            next_sample_time = time.time()
            
            while self.recording:
                current_time = time.time()
                
                # Read from serial port
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) == 9:
                            try:
                                data = list(map(float, parts))
                                
                                with self.data_lock:
                                    csv_writer.writerow([current_time] + data)
                                    csv_file.flush()
                                    
                                    # Also store in memory for HDF5
                                    self.imu_timestamps.append(current_time)
                                    self.imu_data.append(data)
                                    
                            except ValueError:
                                continue  # Skip invalid data
                
                # Wait until next sample time
                next_sample_time += self.imu_interval
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            print(f"IMU recording error: {e}")
        finally:
            csv_file.close()
            print("IMU recording stopped")

    def record_video(self):
        print(f"Video recording started at {self.camera_frequency} FPS...")
        try:
            next_frame_time = time.perf_counter() 
            frames_grabbed = 0
            t0 = time.perf_counter()

            while self.recording:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)

                ret, frame = self.cap.read()
                next_frame_time += self.camera_interval 

                if not ret:
                    print("Failed to read frame from webcam")
                    continue  

                with self.data_lock:
                    self.video_writer.write(frame)
                    self.video_frames.append(frame.copy())
                    self.video_timestamps.append(time.time())

                frames_grabbed += 1

            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                print(f"Achieved video FPS: {frames_grabbed/elapsed:.2f}")

        except Exception as e:
            print(f"Video recording error: {e}")
        finally:
            print("Video recording stopped")

    def save_to_hdf5(self):
        """Save collected data to HDF5 format and clean up temporary files"""
        print("Saving data to HDF5 format...")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_hdf5), exist_ok=True)
            
            # Convert lists to numpy arrays
            imu_timestamps = np.array(self.imu_timestamps)
            imu_data = np.array(self.imu_data)
            video_timestamps = np.array(self.video_timestamps)
            video_frames = np.array(self.video_frames)
            
            print(f"IMU data: {len(imu_timestamps)} samples")
            print(f"Video data: {len(video_frames)} frames")
            
            # Write to HDF5
            with h5py.File(self.output_hdf5, 'w') as hf:
                # Create groups
                imu_group = hf.create_group('imu')
                video_group = hf.create_group('video')
                
                # Write IMU data
                imu_group.create_dataset('timestamps', data=imu_timestamps, compression='gzip')
                imu_group.create_dataset('data', data=imu_data, compression='gzip')
                imu_group.attrs['description'] = 'IMU sensor data: [ax, ay, az, gx, gy, gz, mx, my, mz]'
                imu_group.attrs['frequency'] = self.imu_frequency
                imu_group.attrs['units'] = 'accelerometer: m/s², gyroscope: rad/s, magnetometer: µT'
                
                # Write video data
                video_group.create_dataset('timestamps', data=video_timestamps, compression='gzip')
                video_group.create_dataset('frames', data=video_frames, compression='gzip')
                video_group.attrs['description'] = 'Video frames from webcam'
                video_group.attrs['fps'] = self.camera_frequency
                video_group.attrs['resolution'] = f'{self.camera_width}x{self.camera_height}'
                video_group.attrs['color_format'] = 'BGR'
                
                # Add metadata
                hf.attrs['imu_frequency'] = self.imu_frequency
                hf.attrs['camera_frequency'] = self.camera_frequency
                hf.attrs['imu_samples'] = len(imu_timestamps)
                hf.attrs['video_frames'] = len(video_frames)
                hf.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"HDF5 file saved to: {self.output_hdf5}")
            
            # Clean up temporary files
            self.cleanup_temp_files()
            
            print("Data saving completed successfully!")
            
        except Exception as e:
            print(f"Error during HDF5 conversion: {e}")
            print("Temporary files preserved due to conversion error")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        print("Cleaning up temporary files...")
        if os.path.exists(self.tmp_output_csv):
            os.remove(self.tmp_output_csv)
            print(f"Deleted: {self.tmp_output_csv}")
        
        if os.path.exists(self.tmp_output_mp4):
            os.remove(self.tmp_output_mp4)
            print(f"Deleted: {self.tmp_output_mp4}")
    
    def start_recording(self, duration=15):
        """Start recording for specified duration"""
        print(f"Starting dual recording (IMU + Video) for {duration} seconds...")
        print("Recording will automatically stop after the specified duration")
        print("Press Ctrl+C to stop recording early if needed")
        
        # Initialize hardware
        if not self.initialize_hardware():
            print("Failed to initialize hardware. Exiting.")
            return False
        
        # Start recording
        self.recording = True
        self.start_time = time.time()
        
        # Create and start threads
        self.imu_thread = threading.Thread(target=self.record_imu_data)
        self.video_thread = threading.Thread(target=self.record_video)
        
        self.imu_thread.start()
        self.video_thread.start()
        
        try:
            # Keep main thread alive for specified duration
            while self.recording:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Check if duration has passed
                if elapsed_time >= duration:
                    print(f"\n{duration} seconds elapsed. Stopping recording...")
                    self.recording = False
                    break
                
                # Show progress every 1 second
                if int(elapsed_time) % 1 == 0 and elapsed_time > 0:
                    remaining = duration - elapsed_time
                    print(f"Recording... {elapsed_time:.1f}s elapsed, {remaining:.1f}s remaining")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nRecording stopped by user (Ctrl+C)")
            self.recording = False
        
        # Wait for threads to finish
        print("Waiting for threads to finish...")
        self.imu_thread.join()
        self.video_thread.join()
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        expected_imu_samples = int(self.imu_frequency * total_time)
        expected_video_frames = int(self.camera_frequency * total_time)
        
        print("Recording completed!")
        print(f"Total recording time: {total_time:.2f} seconds")
        print(f"Expected IMU samples: ~{expected_imu_samples}")
        print(f"Expected video frames: ~{expected_video_frames}")
        print(f"Actual IMU samples: {len(self.imu_timestamps)}")
        print(f"Actual video frames: {len(self.video_frames)}")
        
        # Save to HDF5
        self.save_to_hdf5()
        
        # Clean up hardware
        self.cleanup_hardware()
        
        return True


def main():
    """Main function to run the data collection"""
    # Create data collector instance
    collector = DataCollector(
        serial_port="/dev/ttyACM0",
        baud_rate=115200,
        webcam_port=0,
        imu_freq=200,  # 200 Hz
        camera_freq=30  # 30 FPS
    )
    
    # Start recording for 15 seconds
    success = collector.start_recording(duration=15)
    
    if success:
        print("Data collection completed successfully!")
    else:
        print("Data collection failed!")

if __name__ == "__main__":
    main()