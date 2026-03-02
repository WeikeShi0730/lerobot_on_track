#!/usr/bin/env python3
"""
Test script for LeRobot OpenCV camera - Dual Camera Version (Robust)
Displays feeds from two cameras side by side with auto-configuration
"""

import cv2
import time
import platform
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def try_connect_camera(device_path, label="Camera", rotation=Cv2Rotation.NO_ROTATION):
    """Try to connect to camera with fallback configurations"""
    
    # Configuration attempts in order of preference
    configs = [
        # Try with default settings (no explicit width/height/fps)
        {
            "index_or_path": device_path,
            "color_mode": ColorMode.RGB,
            "rotation": rotation
        },
        # Try with 640x480 @ 30fps
        {
            "index_or_path": device_path,
            "fps": 30,
            "width": 640,
            "height": 480,
            "color_mode": ColorMode.RGB,
            "rotation": rotation
        },
        # Try with 640x480 @ 15fps
        {
            "index_or_path": device_path,
            "fps": 15,
            "width": 640,
            "height": 480,
            "color_mode": ColorMode.RGB,
            "rotation": rotation
        },
        # Try with 320x240 @ 30fps (lower resolution)
        {
            "index_or_path": device_path,
            "fps": 30,
            "width": 320,
            "height": 240,
            "color_mode": ColorMode.RGB,
            "rotation": rotation
        },
    ]
    
    for i, config_dict in enumerate(configs):
        try:
            config = OpenCVCameraConfig(**config_dict)
            camera = OpenCVCamera(config)
            camera.connect()
            
            # Try to read a test frame to verify it works
            test_frame = camera.async_read(timeout_ms=2000)
            if test_frame is not None:
                print(f"✓ {label} connected successfully!")
                print(f"  Config: {config.width if hasattr(config, 'width') else 'auto'}x{config.height if hasattr(config, 'height') else 'auto'} @ {config.fps if hasattr(config, 'fps') else 'auto'}fps")
                return camera, config, test_frame.shape
            else:
                camera.disconnect()
                print(f"  Attempt {i+1}: Camera opened but no frame received")
        except Exception as e:
            print(f"  Attempt {i+1} failed: {str(e)[:100]}")
            continue
    
    return None, None, None


def main():
    print("LeRobot OpenCV Dual Camera Test (Robust)")
    print("=" * 60)
    system = platform.system()
    print(f"Platform: {system}")
    
    # Camera device paths (from lerobot-find-cameras)
    camera_path_1 = "/dev/video1"
    camera_path_2 = "/dev/video3"
    
    print(f"\nCamera Configuration:")
    print(f"  Camera 1 Path: {camera_path_1}")
    print(f"  Camera 2 Path: {camera_path_2}")
    
    # Connect to first camera
    print(f"\nConnecting to camera 1 ({camera_path_1})...")
    camera1, config1, shape1 = try_connect_camera(camera_path_1, "Camera 1", rotation=Cv2Rotation.NO_ROTATION)
    
    if camera1 is None:
        print(f"\n✗ Failed to connect to camera 1")
        print("Troubleshooting:")
        print("1. Check camera permissions: ls -l /dev/video*")
        print("2. Add your user to video group: sudo usermod -a -G video $USER")
        print("3. Try running with sudo: sudo python test_dual_camera.py")
        print("4. Check if camera is in use: lsof /dev/video1")
        return
    
    # Connect to second camera (with 180 degree rotation since it's upside down)
    print(f"\nConnecting to camera 2 ({camera_path_2})...")
    camera2, config2, shape2 = try_connect_camera(camera_path_2, "Camera 2", rotation=Cv2Rotation.ROTATE_180)
    
    if camera2 is None:
        print(f"\n✗ Failed to connect to camera 2")
        print("Will continue with single camera for testing...")
        camera1.disconnect()
        return
    
    # Determine display dimensions
    height = shape1[0]  # Use first camera's height
    width = shape1[1]
    
    print(f"\nDisplay configuration:")
    print(f"  Frame size: {width}x{height}")
    print(f"  Combined size: {width*2}x{height}")
    
    # Create display window
    window_name = "Dual Camera View (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
            
            # Resize frames if they don't match (in case cameras have different resolutions)
            if frame1.shape != frame2.shape:
                target_height = min(frame1.shape[0], frame2.shape[0])
                target_width = min(frame1.shape[1], frame2.shape[1])
                frame1 = cv2.resize(frame1, (target_width, target_height))
                frame2 = cv2.resize(frame2, (target_width, target_height))
            
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