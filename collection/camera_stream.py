#!/usr/bin/env python3
"""
Camera Stream and Visualization Program
Streams and displays live camera feed from webcam
"""

import cv2
import argparse
import sys
import time


def stream_camera(camera_port=0, width=640, height=480, fps=30, show_fps=True):
    """
    Stream and visualize camera feed
    
    Args:
        camera_port: Camera port number (default: 0)
        width: Frame width (default: 640)
        height: Frame height (default: 480)
        fps: Target FPS (default: 30)
        show_fps: Whether to display FPS on frame (default: True)
    """
    print(f"Initializing camera on port {camera_port}...")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_port)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera on port {camera_port}")
        print("Try a different port number (e.g., 1, 2) if you have multiple cameras")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized successfully!")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps}")
    print("\n" + "="*50)
    print("Camera streaming started!")
    print("Press 'q' or ESC to quit")
    print("="*50 + "\n")
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:  # Update FPS every second
                fps_display = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Draw FPS on frame
            if show_fps:
                fps_text = f"FPS: {fps_display:.1f}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw resolution info
            resolution_text = f"{actual_width}x{actual_height}"
            cv2.putText(frame, resolution_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Camera Stream', frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nStopping camera stream...")
                break
                
    except KeyboardInterrupt:
        print("\nStopping camera stream (KeyboardInterrupt)...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stream and visualize camera feed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python camera_stream.py
  python camera_stream.py --port 1
  python camera_stream.py --width 1280 --height 720
  python camera_stream.py --port 0 --fps 60 --no-fps
        '''
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=0,
        help='Camera port number (default: 0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Frame width (default: 640)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Frame height (default: 480)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target FPS (default: 30)'
    )
    
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Hide FPS display on frame'
    )
    
    args = parser.parse_args()
    
    success = stream_camera(
        camera_port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show_fps=not args.no_fps
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

