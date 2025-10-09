# BATCH BACKGROUND REMOVAL + THRESHOLDING (B&W OUTPUT)
# Removes background and TH multiple avi files in a folder
# Folder needs to be specified within script TO BE MODIFIED
# Best codec for .avi: IYUV or MJPG

import cv2
import numpy as np
import os
from glob import glob

def remove_background_from_video(video_path, output_path, apply_gaussian=True, ksize=(5, 5), threshold_value=3):
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # --- First pass: compute average background ---
    avg_frame = np.zeros((height, width, 3), dtype=np.float64)
    count = 0

    print(f"Computing background for: {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        avg_frame += frame.astype(np.float64)
        count += 1

    cap.release()

    if count > 0:
        avg_frame /= count
    avg_frame = avg_frame.astype(np.uint8)

    # --- Second pass: subtract background, threshold, and save video ---
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # good general-purpose codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"Processing and saving thresholded output: {os.path.basename(output_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Subtract background
        fg = cv2.absdiff(frame, avg_frame)

        if apply_gaussian:
            fg = cv2.GaussianBlur(fg, ksize, 0)

        # Convert to grayscale
        gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get black & white result
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Write to output
        out.write(thresh)

    cap.release()
    out.release()
    print(f"Done: {os.path.basename(video_path)} → {os.path.basename(output_path)}\n")


def process_folder(input_folder, output_folder, apply_gaussian=True, threshold_value=3):
    os.makedirs(output_folder, exist_ok=True)
    video_files = glob(os.path.join(input_folder, "*.avi"))

    for video_file in video_files:
        filename = os.path.basename(video_file)
        output_file = os.path.join(output_folder, f"processed_{filename}")
        remove_background_from_video(video_file, output_file, apply_gaussian, threshold_value=threshold_value)


if __name__ == "__main__":
    input_folder = "/Users/jorgelunaherrera/Documents/BHV 22SEP2025/"
    output_folder = "/Users/jorgelunaherrera/Documents/BHV 22SEP2025/processed/"

    process_folder(input_folder, output_folder, apply_gaussian=True, threshold_value=5)
