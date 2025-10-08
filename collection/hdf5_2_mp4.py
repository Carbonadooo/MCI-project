import h5py
import cv2
import os
from pathlib import Path

def hdf5_to_mp4(input_hdf5_path):
    """Convert HDF5 video data to MP4 format and save in same directory"""
    
    print(f"Converting HDF5 to MP4: {input_hdf5_path}")
    
    # Validate input file
    if not os.path.exists(input_hdf5_path):
        raise FileNotFoundError(f"Input file not found: {input_hdf5_path}")
    
    # Generate output path in same directory
    input_path = Path(input_hdf5_path)
    output_mp4_path = input_path.parent / f"{input_path.stem}.mp4"
    
    try:
        # Open HDF5 file
        with h5py.File(input_hdf5_path, 'r') as hf:
            print("Reading video data from HDF5...")
            
            # Check if video data exists
            if 'video/frames' not in hf:
                raise ValueError("No video data found in HDF5 file")
            
            # Load video data
            video_frames = hf['video/frames'][:]
            
            # Get metadata
            video_attrs = dict(hf['video'].attrs)
            
            print(f"Loaded {len(video_frames)} video frames")
            print(f"Resolution: {video_frames.shape[1]}x{video_frames.shape[2]}")
            
            # Use original FPS or default to 30
            fps = video_attrs.get('fps', 30)
            print(f"FPS: {fps}")
            
            # Get video dimensions
            height, width = video_frames.shape[1], video_frames.shape[2]
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise RuntimeError("Failed to initialize video writer")
            
            print("Writing video frames...")
            
            # Write frames
            for i, frame in enumerate(video_frames):
                out.write(frame)
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / len(video_frames) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(video_frames)} frames)")
            
            # Release video writer
            out.release()
            
            print(f"✅ Video conversion completed!")
            print(f"Output file: {output_mp4_path}")
            print(f"Duration: {len(video_frames) / fps:.2f} seconds")
            
            return str(output_mp4_path)
            
    except Exception as e:
        print(f"❌ Error converting HDF5 to MP4: {e}")
        raise

if __name__ == "__main__":
    # Convert the specified HDF5 file
    input_file = "/home/chuye/Documents/MCI-project/data/output_data.hdf5"
    
    try:
        output_path = hdf5_to_mp4(input_file)
        print(f"\n🎉 Successfully converted to: {output_path}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")