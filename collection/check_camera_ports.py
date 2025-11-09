#!/usr/bin/env python3
"""
Camera Port Detection Utility
Helps identify available camera ports on the system
"""

import cv2
import os
import glob
import sys


def check_video_devices():
    """Check for video devices in /dev/video*"""
    video_devices = glob.glob('/dev/video*')
    video_devices.sort()
    return video_devices


def test_camera_port(port):
    """Test if a camera port is accessible"""
    cap = cv2.VideoCapture(port)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return True, width, height, fps
        cap.release()
    return False, None, None, None


def find_available_cameras(max_ports=10):
    """Find all available camera ports"""
    print("="*60)
    print("Camera Port Detection")
    print("="*60)
    print()
    
    # Check /dev/video* devices
    video_devices = check_video_devices()
    if video_devices:
        print(f"Found {len(video_devices)} video device(s) in /dev/:")
        for device in video_devices:
            print(f"  - {device}")
        print()
    else:
        print("No /dev/video* devices found")
        print()
    
    print("Testing camera ports (this may take a few seconds)...")
    print()
    
    available_cameras = []
    
    # Test common port numbers
    for port in range(max_ports):
        print(f"Testing port {port}...", end=" ")
        is_available, width, height, fps = test_camera_port(port)
        
        if is_available:
            print(f"✓ AVAILABLE")
            print(f"    Resolution: {width}x{height}")
            print(f"    FPS: {fps}")
            available_cameras.append({
                'port': port,
                'width': width,
                'height': height,
                'fps': fps
            })
        else:
            print("✗ Not available")
    
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    
    if available_cameras:
        print(f"\nFound {len(available_cameras)} available camera(s):\n")
        for cam in available_cameras:
            print(f"  Port {cam['port']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS")
        print()
        print("To use a camera, run:")
        print(f"  python collection/camera_stream.py --port {available_cameras[0]['port']}")
    else:
        print("\n❌ No available cameras found!")
        print("\nTroubleshooting tips:")
        print("  1. Make sure your camera is connected")
        print("  2. Check if the camera is being used by another application")
        print("  3. Try running with sudo (may be a permissions issue)")
        print("  4. Check USB connection if using USB camera")
        print("  5. Run 'lsusb' to see if camera is detected by system")
    
    print("="*60)
    
    return available_cameras


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect available camera ports',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--max-ports',
        type=int,
        default=10,
        help='Maximum port number to test (default: 10)'
    )
    
    args = parser.parse_args()
    
    cameras = find_available_cameras(max_ports=args.max_ports)
    
    return 0 if cameras else 1


if __name__ == "__main__":
    sys.exit(main())

