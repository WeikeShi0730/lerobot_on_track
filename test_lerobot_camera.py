#!/usr/bin/env python3
"""
Test script for LeRobot OpenCV camera (Cross-platform)
Works on Linux, Windows, and macOS
"""

import cv2
import time
import platform
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def main():
    print("LeRobot OpenCV Camera Test")
    print("=" * 60)
    system = platform.system()
    print(f"Platform: {system}")
    
    # Configure camera
    camera_index = 0  # Change this if you have multiple cameras
    
    print(f"\nCamera Configuration:")
    print(f"  Index: {camera_index}")
    print(f"  Resolution: 640x480")
    print(f"  FPS: 15")
    print(f"  Color Mode: RGB")
    
    # Create config without specifying backend - let it auto-detect
    config = OpenCVCameraConfig(
        index_or_path=camera_index,
        fps=15,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
    
    # Create and connect camera
    print(f"\nConnecting to camera {camera_index}...")
    if system == "Windows":
        print("(This may take up to 20 seconds on Windows...)")
    
    try:
        camera = OpenCVCamera(config)
        camera.connect()
        print("✓ Camera connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect to camera: {e}")
        print("\nTroubleshooting:")
        print("1. Run: python -m lerobot.find_cameras opencv")
        print("2. Try different camera indices (0, 1, 2...)")
        print("3. Make sure no other app is using the camera")
        
        # Try to list available cameras manually
        print("\nManual camera detection:")
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"  ✓ Camera {i} is available")
                    cap.release()
                else:
                    print(f"  ✗ Camera {i} not found")
            except:
                print(f"  ✗ Camera {i} error")
        return
    
    # Create display window
    window_name = "LeRobot Camera Test (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.width, config.height)
    
    print("\n📹 Displaying camera feed...")
    print("   Press ESC or 'q' to quit")
    print("   Press 's' to save a snapshot\n")
    
    frame_count = 0
    start_time = time.time()
    snapshot_count = 0
    
    try:
        while True:
            # Read frame from camera
            try:
                frame = camera.async_read(timeout_ms=1000)
            except Exception as e:
                print(f"Error reading frame: {e}")
                continue
            
            if frame is None:
                print("Warning: Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f} | Frames: {frame_count} | Shape: {frame.shape}")
            
            # Add FPS overlay to frame
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame_bgr,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add instructions
            cv2.putText(
                frame_bgr,
                "ESC/Q: Quit | S: Save snapshot",
                (10, frame_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Display frame
            cv2.imshow(window_name, frame_bgr)
            
            # Check for keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("\nExiting...")
                break
            elif key == ord('s'):  # Save snapshot
                filename = f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(filename, frame_bgr)
                print(f"✓ Saved {filename}")
                snapshot_count += 1
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed")
                    break
            except:
                pass  # Window property check may fail on some systems
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        camera.disconnect()
        cv2.destroyAllWindows()
        print("✓ Done!")
        
        # Print statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nStatistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        if snapshot_count > 0:
            print(f"  Snapshots saved: {snapshot_count}")


if __name__ == "__main__":
    main()