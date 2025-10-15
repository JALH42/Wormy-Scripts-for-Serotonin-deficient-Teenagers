#!/usr/bin/env python3
"""
Video Tracker Plotter for Mac OS - OPTIMIZED WITHOUT THREADING
"""

import os
import glob
import csv
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

class VideoTrackerPlotter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
    def select_folder(self):
        """Open folder selection dialog"""
        print("Please select the folder containing AVI videos...")
        folder_path = filedialog.askdirectory(title="Select Folder with AVI Videos")
        return folder_path
    
    def find_avi_files(self, folder_path):
        """Find all AVI files in the given folder"""
        avi_pattern = os.path.join(folder_path, "*.avi")
        avi_files = glob.glob(avi_pattern)
        return avi_files
    
    def find_csv_files(self, results_folder):
        """Find all CSV files in the results folder"""
        csv_pattern = os.path.join(results_folder, "*.csv")
        csv_files = glob.glob(csv_pattern)
        return csv_files
    
    def load_csv_data(self, csv_file):
        """Load CSV data - OPTIMIZED VERSION"""
        try:
            # Read entire file at once for speed
            with open(csv_file, 'r') as f:
                content = f.read().strip().split('\n')
            
            # Remove first row and split into rows
            rows = [line.split(',') for line in content[1:]]  # Skip header
            
            if len(rows) < 2:
                return None, None
            
            # Vectorized conversion (much faster than loops)
            x_coords = np.array([float(x) if x.strip() else np.nan for x in rows[0]])
            y_coords = np.array([float(y) if y.strip() else np.nan for y in rows[1]])
            
            return x_coords, y_coords
            
        except Exception as e:
            print(f"Error loading CSV {csv_file}: {e}")
            return None, None
    
    def generate_color_map(self, n_colors):
        """Generate distinct colors for each CSV file"""
        colors = cm.rainbow(np.linspace(0, 1, n_colors))
        return [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    
    def create_legend_image(self, csv_files, colors, output_path):
        """Create a JPG legend showing colors and CSV names"""
        # Use non-interactive backend to avoid threading issues
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, max(6, len(csv_files) * 0.5)))
        
        for i, (csv_file, color) in enumerate(zip(csv_files, colors)):
            csv_name = os.path.basename(csv_file).replace('.csv', '')
            bgr_color = [c/255 for c in color[::-1]]  # Convert BGR to RGB for matplotlib
            ax.plot([], [], 'o', color=bgr_color, markersize=10, label=csv_name)
        
        ax.legend(loc='center', fontsize=12, frameon=True, fancybox=True, 
                 shadow=True, facecolor='white')
        ax.set_title('Tracking Data Legend', fontsize=16, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Legend saved to: {output_path}")
    
    def optimize_trajectory_drawing(self, frame, trajectory, color, current_point):
        """Optimized trajectory drawing with batch operations"""
        if len(trajectory) < 2:
            return
        
        # Batch draw lines using polylines (much faster than individual line calls)
        points = np.array(trajectory, dtype=np.int32)
        
        # Draw all lines at once using polylines
        if len(points) >= 2:
            pts = points.reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, color, 2, lineType=cv2.LINE_AA)
        
        # Draw current point
        if current_point is not None:
            cv2.circle(frame, current_point, 3, color, -1)
            cv2.circle(frame, current_point, 4, (255, 255, 255), 1)
    
    def process_video(self, avi_file):
        """Process a single AVI file - OPTIMIZED VERSION"""
        print(f"\nProcessing: {avi_file}")
        
        # Find corresponding frames folder
        base_name = os.path.splitext(os.path.basename(avi_file))[0]
        parent_dir = os.path.dirname(avi_file)
        frames_folder = os.path.join(parent_dir, f"{base_name}_frames")
        results_folder = os.path.join(frames_folder, "results")
        
        if not os.path.exists(results_folder):
            print(f"Results folder not found: {results_folder}")
            return
        
        # Find CSV files
        csv_files = self.find_csv_files(results_folder)
        if not csv_files:
            print(f"No CSV files found in: {results_folder}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load all CSV data
        print("Loading CSV data...")
        tracking_data = []
        valid_csv_files = []
        
        for csv_file in csv_files:
            x_coords, y_coords = self.load_csv_data(csv_file)
            if x_coords is not None and y_coords is not None:
                tracking_data.append((x_coords, y_coords))
                valid_csv_files.append(csv_file)
                print(f"  Loaded {os.path.basename(csv_file)}: {len(x_coords)} frames")
        
        if not tracking_data:
            print("No valid tracking data found")
            return
        
        # Generate colors
        colors = self.generate_color_map(len(valid_csv_files))
        
        # Create legend (no threading)
        legend_path = os.path.join(results_folder, f"{base_name}_legend.jpg")
        self.create_legend_image(valid_csv_files, colors, legend_path)
        
        # Open video
        cap = cv2.VideoCapture(avi_file)
        if not cap.isOpened():
            print(f"Error opening video: {avi_file}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Find minimum number of frames between video and tracking data
        min_frames = min(total_frames, min(len(x) for x, y in tracking_data))
        print(f"Processing {min_frames} frames...")
        
        # Setup output video
        output_path = os.path.join(results_folder, f"{base_name}_traces.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Try MJPG for better performance
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Pre-allocate arrays and variables for speed
        frame_count = 0
        trajectory_history = [[] for _ in range(len(tracking_data))]
        
        # Precompute valid frames to avoid checks in hot loop
        valid_frames_mask = []
        for x_coords, y_coords in tracking_data:
            mask = ~(np.isnan(x_coords[:min_frames]) | np.isnan(y_coords[:min_frames]))
            valid_frames_mask.append(mask)
        
        print("Starting video processing...")
        start_time = cv2.getTickCount()
        
        while frame_count < min_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process all tracks for this frame
            for i, (x_coords, y_coords) in enumerate(tracking_data):
                if valid_frames_mask[i][frame_count]:
                    x, y = x_coords[frame_count], y_coords[frame_count]
                    x_int, y_int = int(x), int(y)
                    current_point = (x_int, y_int)
                    
                    # Add current point to trajectory history
                    trajectory_history[i].append(current_point)
                    
                    # Use optimized drawing
                    self.optimize_trajectory_drawing(
                        frame, trajectory_history[i], colors[i], current_point
                    )
            
            # Add text overlays (minimal for performance)
            if frame_count % 100 == 0:  # Only add text every 100 frames for performance
                cv2.putText(frame, f"Frame: {frame_count}/{min_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Tracks: {len(tracking_data)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
            
            # Progress reporting
            if frame_count % 500 == 0:
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                fps_processed = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{min_frames} frames ({fps_processed:.1f} fps)")
        
        # Calculate final performance
        total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        processing_fps = min_frames / total_time if total_time > 0 else 0
        
        cap.release()
        out.release()
        
        print(f"Finished in {total_time:.1f}s ({processing_fps:.1f} fps)")
        print(f"Output saved to: {output_path}")
        
        # Print final summary
        print("\nFinal tracking data summary:")
        for i, (x_coords, y_coords) in enumerate(tracking_data):
            valid_points = np.sum(valid_frames_mask[i])
            print(f"Track {i}: {valid_points}/{min_frames} valid points, {len(trajectory_history[i])} points plotted")
    
    def run(self):
        """Main execution function"""
        folder_path = self.select_folder()
        
        if not folder_path:
            print("No folder selected. Exiting.")
            return
        
        print(f"Selected folder: {folder_path}")
        
        # Find all AVI files
        avi_files = self.find_avi_files(folder_path)
        
        if not avi_files:
            print("No AVI files found in the selected folder.")
            return
        
        print(f"Found {len(avi_files)} AVI files")
        
        # Process each AVI file
        for avi_file in avi_files:
            self.process_video(avi_file)
        
        print("\nAll videos processed successfully!")
        self.root.destroy()

def main():
    """Main function with error handling"""
    try:
        plotter = VideoTrackerPlotter()
        plotter.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
