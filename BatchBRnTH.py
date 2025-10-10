# BATCH BACKGROUND REMOVAL + THRESHOLDING (B&W OUTPUT, OPTIMIZED)
# Best codec for .avi: IYUV or MJPG

# Removes background and thresholds multiple .avi files in a selected folder.
# At runtime, a window will prompt you to choose the input folder.
# The output will be written to a 'processed' subfolder inside the chosen input folder.

import cv2
import numpy as np
import os
import sys
from glob import glob

# GUI for selecting input directory
import tkinter as tk
from tkinter import filedialog, messagebox


def compute_background(video_path):
    """Compute the average background using cv2.accumulate (fast C++)."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    height, width = frame.shape[:2]
    avg_frame = np.zeros((height, width, 3), dtype=np.float32)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.accumulate(frame, avg_frame)
        count += 1

    cap.release()

    if count > 0:
        avg_frame /= count

    return avg_frame.astype(np.uint8)


def remove_background_from_video(video_path, output_path, apply_gaussian=True, ksize=(5, 5), threshold_value=3):
    print(f"Computing background for: {os.path.basename(video_path)}")
    avg_frame = compute_background(video_path)
    if avg_frame is None:
        print(f"Skipping {video_path} (could not read frames).")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"Processing and saving thresholded output: {os.path.basename(output_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg = cv2.absdiff(frame, avg_frame)

        if apply_gaussian:
            fg = cv2.GaussianBlur(fg, ksize, 0)

        gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        out.write(thresh)

    cap.release()
    out.release()
    print(f"Done: {os.path.basename(video_path)} → {os.path.basename(output_path)}\n")


def process_folder(input_folder, output_folder, apply_gaussian=True, threshold_value=3):
    os.makedirs(output_folder, exist_ok=True)
    video_files = glob(os.path.join(input_folder, "*.avi"))

    if not video_files:
        print(f"No .avi files found in: {input_folder}")
        return

    for video_file in video_files:
        filename = os.path.basename(video_file)
        output_file = os.path.join(output_folder, f"processed_{filename}")
        remove_background_from_video(video_file, output_file, apply_gaussian, threshold_value=threshold_value)


def select_input_folder_dialog():
    """Open a folder selection dialog and return the chosen path, or '' if cancelled."""
    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the front on some platforms
    root.update()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Select input folder containing .avi videos")
    root.attributes("-topmost", False)
    root.destroy()
    return selected


if __name__ == "__main__":
    input_folder = select_input_folder_dialog()
    if not input_folder:
        print("No folder selected. Exiting.")
        sys.exit(0)

    output_folder = os.path.join(input_folder, "processed")

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Adjust threshold_value if needed
    process_folder(input_folder, output_folder, apply_gaussian=True, threshold_value=5)
