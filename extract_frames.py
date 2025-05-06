import os
import cv2
import argparse

def extract_frames(video_path, output_dir, target_fps=24):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_dir = os.path.join(output_dir, video_name)

    # Create directory if it doesn't exist
    os.makedirs(frame_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Could not determine FPS of the video.")
        return

    frame_interval = int(round(original_fps / target_fps))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(frame_output_dir, filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to: {frame_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames at 24 fps from a video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir)
