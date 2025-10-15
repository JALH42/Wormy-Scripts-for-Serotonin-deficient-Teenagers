#!/usr/bin/env python3
"""
Video Tracker Plotter for Mac OS - PLOTS ALL PREVIOUS POINTS
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
        """Load CSV data - CORRECTED VERSION"""
        try:
            print(f"Loading CSV: {csv_file}")
            
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Debug original structure
            print(f"Original CSV structure: {len(rows)} rows, {len(rows[0]) if rows else 0} columns")
            
            # Remove first row as requested
            if len(rows) > 0:
                rows = rows[1:]
            
            if len(rows) < 2:
                print(f"Not enough data rows: {len(rows)}")
                return None, None
            
            # The CSV has 2 rows (X and Y) and many columns (frames)
            # Each column represents one frame
            x_coords = []
            y_coords = []
            
            # Convert each value in the rows
            for i in range(len(rows[0])):  # Iterate through columns (frames)
                try:
                    x_val = float(rows[0][i]) if rows[0][i].strip() else np.nan
                    y_val = float(rows[1][i]) if rows[1][i].strip() else np.nan
                    x_coords.append(x_val)
                    y_coords.append(y_val)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not convert value at column {i}: {e}")
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)
            
            x_array = np.array(x_coords)
            y_array = np.array(y_coords)
            
            print(f"Loaded {len(x_array)} frames")
            print(f"First 3 X values: {x_array[:3]}")
            print(f"First 3 Y values: {y_array[:3]}")
            print(f"Last 3 X values: {x_array[-3:]}")
            print(f"Last 3 Y values: {y_array[-3:]}")
            
            return x_array, y_array
            
        except Exception as e:
            print(f"Error loading CSV {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def generate_color_map(self, n_colors):
        """Generate distinct colors for each CSV file"""
        colors = cm.rainbow(np.linspace(0, 1, n_colors))
        return [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    
    def create_legend_image(self, csv_files, colors, output_path):
        """Create a JPG legend showing colors and CSV names"""
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
    
    def process_video(self, avi_file):
        """Process a single AVI file with its corresponding CSV data"""
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
        tracking_data = []
        valid_csv_files = []
        
        for csv_file in csv_files:
            x_coords, y_coords = self.load_csv_data(csv_file)
            if x_coords is not None and y_coords is not None:
                tracking_data.append((x_coords, y_coords))
                valid_csv_files.append(csv_file)
                print(f"Successfully loaded {os.path.basename(csv_file)} with {len(x_coords)} frames")
        
        if not tracking_data:
            print("No valid tracking data found")
            return
        
        # Generate colors
        colors = self.generate_color_map(len(valid_csv_files))
        
        # Create legend
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
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Find minimum number of frames between video and tracking data
        min_frames = min(total_frames, min(len(x) for x, y in tracking_data))
        print(f"Will process {min_frames} frames (minimum between video and tracking data)")
        
        # Setup output video
        output_path = os.path.join(results_folder, f"{base_name}_traces.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Compressed AVI
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing {min_frames} frames...")
        
        frame_count = 0
        # Store ALL previous points for each track
        trajectory_history = [[] for _ in range(len(tracking_data))]
        
        while frame_count < min_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Plot tracking data for current frame
            for i, (x_coords, y_coords) in enumerate(tracking_data):
                if frame_count < len(x_coords):
                    x, y = x_coords[frame_count], y_coords[frame_count]
                    
                    # Check if coordinates are valid numbers
                    if not np.isnan(x) and not np.isnan(y):
                        x_int, y_int = int(x), int(y)
                        
                        # Debug first few frames
                        if frame_count < 5:
                            print(f"Frame {frame_count}, Track {i}: ({x:.1f}, {y:.1f}) -> ({x_int}, {y_int})")
                        
                        # Add current point to trajectory history (ALL previous points)
                        trajectory_history[i].append((x_int, y_int))
                        
                        # Draw trajectory for ALL previous points
                        for j in range(1, len(trajectory_history[i])):
                            cv2.line(frame, 
                                   trajectory_history[i][j-1], 
                                   trajectory_history[i][j], 
                                   colors[i], 2)
                        
                        # Draw current position with smaller dots
                        cv2.circle(frame, (x_int, y_int), 3, colors[i], -1)
                        cv2.circle(frame, (x_int, y_int), 4, (255, 255, 255), 1)
            
            # Add frame counter and track info
            cv2.putText(frame, f"Frame: {frame_count}/{min_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracks: {len(tracking_data)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Showing ALL previous points", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{min_frames} frames")
                # Show how many points we've accumulated so far
                for i in range(len(trajectory_history)):
                    print(f"  Track {i}: {len(trajectory_history[i])} points accumulated")
        
        cap.release()
        out.release()
        
        print(f"Finished processing. Output saved to: {output_path}")
        
        # Print final summary
        print("\nFinal tracking data summary:")
        for i, (x_coords, y_coords) in enumerate(tracking_data):
            valid_points = np.sum(~np.isnan(x_coords[:min_frames]) & ~np.isnan(y_coords[:min_frames]))
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
