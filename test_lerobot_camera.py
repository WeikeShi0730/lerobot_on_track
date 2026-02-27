#!/usr/bin/env python3
"""
Test script for LeRobot OpenCV camera - Dual Camera Version
Displays feeds from two cameras side by side
"""

import cv2
import time
import platform
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def main():
    print("LeRobot OpenCV Dual Camera Test")
    print("=" * 60)
    system = platform.system()
    print(f"Platform: {system}")
    
    # Configure both cameras
    # Use device paths from lerobot-find-cameras output
    # These are the FFMPEG cameras with actual video streams
    camera_index_1 = "/dev/video1"  # First camera (Camera #0 from scan)
    camera_index_2 = "/dev/video3"  # Second camera (Camera #5 from scan)
    
    # Shared configuration for both cameras
    width = 640
    height = 480
    fps = 15
    
    print(f"\nCamera Configuration:")
    print(f"  Camera 1 Index: {camera_index_1}")
    print(f"  Camera 2 Index: {camera_index_2}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Color Mode: RGB")
    
    # Create configs for both cameras
    config1 = OpenCVCameraConfig(
        index_or_path=camera_index_1,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
    
    config2 = OpenCVCameraConfig(
        index_or_path=camera_index_2,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
    
    # Connect to both cameras
    print(f"\nConnecting to camera {camera_index_1}...")
    try:
        camera1 = OpenCVCamera(config1)
        camera1.connect()
        print("✓ Camera 1 connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect to camera 1: {e}")
        print("\nTroubleshooting:")
        print("1. Run: python -m lerobot.find_cameras opencv")
        print("2. Check camera indices")
        return
    
    print(f"\nConnecting to camera {camera_index_2}...")
    try:
        camera2 = OpenCVCamera(config2)
        camera2.connect()
        print("✓ Camera 2 connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect to camera 2: {e}")
        camera1.disconnect()
        print("\nNote: Only one camera detected. Adjust camera_index_2 if needed.")
        return
    
    # Create display window
    window_name = "Dual Camera View (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Make window twice as wide to fit both cameras side by side
    cv2.resizeWindow(window_name, width * 2, height)
    
    print("\n📹 Displaying dual camera feed...")
    print("   Press ESC or 'q' to quit")
    print("   Press 's' to save snapshots from both cameras\n")
    
    frame_count = 0
    start_time = time.time()
    snapshot_count = 0
    
    # Colors for labels
    label_color = (0, 255, 0)  # Green
    
    try:
        while True:
            # Read frames from both cameras
            try:
                frame1 = camera1.async_read(timeout_ms=1000)
                frame2 = camera2.async_read(timeout_ms=1000)
            except Exception as e:
                print(f"Error reading frames: {e}")
                continue
            
            if frame1 is None or frame2 is None:
                print("Warning: Failed to read from one or both cameras")
                time.sleep(0.1)
                continue
            
            # Convert RGB to BGR for OpenCV display
            frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            if frame_count % 30 == 0:
                print(f"FPS: {current_fps:.1f} | Frames: {frame_count}")
            
            # Add labels and FPS to each frame
            # Camera 1
            cv2.putText(
                frame1_bgr,
                "Camera 1",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                label_color,
                2
            )
            cv2.putText(
                frame1_bgr,
                f"FPS: {current_fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                label_color,
                2
            )
            
            # Camera 2
            cv2.putText(
                frame2_bgr,
                "Camera 2",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                label_color,
                2
            )
            cv2.putText(
                frame2_bgr,
                f"FPS: {current_fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                label_color,
                2
            )
            
            # Add instructions at bottom of first frame
            cv2.putText(
                frame1_bgr,
                "ESC/Q: Quit | S: Save",
                (10, frame1_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Concatenate frames horizontally (side by side)
            combined_frame = np.hstack([frame1_bgr, frame2_bgr])
            
            # Display combined frame
            cv2.imshow(window_name, combined_frame)
            
            # Check for keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("\nExiting...")
                break
            elif key == ord('s'):  # Save snapshots
                filename1 = f"camera1_snapshot_{snapshot_count:04d}.png"
                filename2 = f"camera2_snapshot_{snapshot_count:04d}.png"
                filename_combined = f"combined_snapshot_{snapshot_count:04d}.png"
                
                cv2.imwrite(filename1, frame1_bgr)
                cv2.imwrite(filename2, frame2_bgr)
                cv2.imwrite(filename_combined, combined_frame)
                
                print(f"✓ Saved {filename1}, {filename2}, and {filename_combined}")
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
        camera1.disconnect()
        camera2.disconnect()
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
            print(f"  Snapshots saved: {snapshot_count} (from each camera)")


if __name__ == "__main__":
    main()