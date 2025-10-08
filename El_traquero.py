#!/usr/bin/env python3
"""
El Traquero - Multi-Worm Tracking System
========================================
A sophisticated system for tracking multiple worms across video frames.

El Traquero (Spanish for "The Tracker") automatically extracts frames from AVI videos,
allows interactive worm selection, and tracks worm trajectories with customizable parameters.

Authors: 
Jorge Alejandro Luna Herrera (a.k.a. Mr. Gusanos)
Natalia Soledad Lafourcade Luna Herrera (a.k.a. Nata, Natilla)
Sputnik Gregorio El Mar Luna Herrera (a.k.a. Goyo, Goyito)

Remastered by: 
Deepseek

Date: [Current Date]
Version: 3.0 - Optimized
"""

import os
import cv2
from scipy.ndimage import label, center_of_mass
import numpy as np
from numpy import savetxt
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for stability
import matplotlib.pyplot as plt

# Try to import Numba, but provide fallback if it causes issues
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✅ Numba JIT available - using optimized functions")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not available - using slower Python functions")

# =============================================================================
# USER CONFIGURATION SECTION - CUSTOMIZE THESE PARAMETERS FOR YOUR EXPERIMENT
# =============================================================================

# INPUT/OUTPUT PATHS (Update these for your system)
VIDEO_PATH = "/Users/jorgelunaherrera/Documents/Jorge - Tracking sample/i2.avi"          # Path to your input AVI video file
FRAMES_DIR = "/Users/jorgelunaherrera/Documents/Jorge - Tracking sample/extracted_framesi2"         # Directory where frames will be extracted
OUTPUT_DIR = "/Users/jorgelunaherrera/Documents/Jorge - Tracking sample/extracted_framesi2/results" # Directory for output files and results

# TRACKING PARAMETERS (Adjust these based on your video characteristics)
MIN_BLOB_SIZE = 25                        # Minimum particle size to consider as worm (pixels)
MAX_DISTANCE_LIMIT = 50                   # Maximum distance for worm matching between frames (pixels)
EXTRACT_ALL_FRAMES = True                 # Set to False to extract only first N frames for testing
MAX_FRAMES_TO_EXTRACT = 1000              # Only used if EXTRACT_ALL_FRAMES is False

# PERFORMANCE OPTIMIZATION PARAMETERS
USE_MULTIPROCESSING = False               # Experimental: Use multiprocessing for faster processing (True/False)
FRAME_SKIP = 1                            # Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)
PRELOAD_FRAMES = False                    # Preload all frames to memory (faster but uses more RAM)

# IMAGE PROCESSING PARAMETERS
BACKGROUND_SUBTRACTION = False            # Experimental: Use background subtraction (True/False)
GAUSSIAN_BLUR_SIZE = 0                    # Size of Gaussian blur kernel (0 to disable blurring)
BINARY_THRESHOLD = 25                    # Threshold for binary conversion (0-255)

# VISUALIZATION PARAMETERS
PLOT_COLORS = ['tab:cyan', 'tab:olive', 'tab:pink', 'tab:brown', 'tab:purple', 
               'tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:gray',
               'yellow', 'lightblue', 'lightgreen', 'violet', 'gold']
PLOT_BACKGROUND_COLOR = 'black'           # Background color for trajectory plot
PLOT_LINE_WIDTH = 2                       # Line width for trajectory lines
PLOT_DPI = 150                            # Resolution for saved plot (150 for speed, 300 for high quality)

# ADVANCED PARAMETERS (Modify only if you understand the consequences)
DISTANCE_CALCULATION = 'squared'          # 'squared' for speed, 'euclidean' for accuracy
WORM_MATCHING_STRATEGY = 'greedy'         # 'greedy' for speed, 'optimal' for accuracy

# =============================================================================
# END OF USER CONFIGURATION - ADVANCED USERS CAN MODIFY BELOW THIS LINE
# =============================================================================

# Start timing for performance monitoring
start_time = time.time()

# Numba-optimized function with error handling
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)  # cache=True for faster subsequent runs
    def ParticleSizeFilt(TagParticles, NumParticles, image, min_blob_size):
        """
        Filter out small particles (noise) from the binary image using Numba JIT for speed.
        
        Parameters:
        -----------
        TagParticles : numpy.ndarray
            Labeled array where each connected component has a unique integer value
        NumParticles : int
            Total number of labeled particles in the image
        image : numpy.ndarray
            Original grayscale image to be filtered in-place
        min_blob_size : int
            Minimum size in pixels for a blob to be considered a worm
        
        Returns:
        --------
        numpy.ndarray
            Filtered image with small particles removed
        """
        # Initialize dictionary to store particle sizes
        SizeDic = {}
        max_particle = TagParticles.max()
        for i in range(max_particle):
            SizeDic[i + 1] = 0

        # Count pixels for each particle
        for j in range(TagParticles.shape[0]):
            for k in range(TagParticles.shape[1]):
                pixel_val = TagParticles[j, k]
                if pixel_val > 0:
                    SizeDic[pixel_val] += 1

        # Identify small particles (potential noise)
        SmallParticles = set()
        for k, v in SizeDic.items():
            if v < min_blob_size:
                SmallParticles.add(k)

        # Remove small particles from the image
        for y42 in range(TagParticles.shape[0]):
            for z42 in range(TagParticles.shape[1]):
                pixel = TagParticles[y42, z42]
                if pixel in SmallParticles:
                    image[y42, z42] = 0  # Set pixel to black

        return image
else:
    def ParticleSizeFilt(TagParticles, NumParticles, image, min_blob_size):
        """
        Fallback function without Numba optimization.
        
        Parameters:
        -----------
        TagParticles : numpy.ndarray
            Labeled array where each connected component has a unique integer value
        NumParticles : int
            Total number of labeled particles in the image
        image : numpy.ndarray
            Original grayscale image to be filtered in-place
        min_blob_size : int
            Minimum size in pixels for a blob to be considered a worm
        
        Returns:
        --------
        numpy.ndarray
            Filtered image with small particles removed
        """
        # Initialize dictionary to store particle sizes
        SizeDic = {}
        max_particle = TagParticles.max()
        for i in range(max_particle):
            SizeDic[i + 1] = 0

        # Count pixels for each particle
        for j in range(TagParticles.shape[0]):
            for k in range(TagParticles.shape[1]):
                pixel_val = TagParticles[j, k]
                if pixel_val > 0:
                    SizeDic[pixel_val] += 1

        # Identify small particles (potential noise)
        SmallParticles = set()
        for k, v in SizeDic.items():
            if v < min_blob_size:
                SmallParticles.add(k)

        # Remove small particles from the image
        for y42 in range(TagParticles.shape[0]):
            for z42 in range(TagParticles.shape[1]):
                pixel = TagParticles[y42, z42]
                if pixel in SmallParticles:
                    image[y42, z42] = 0  # Set pixel to black

        return image

def extract_frames_from_video(video_path, output_dir, max_frames=None):
    """
    Extract individual frames from an AVI video file and save as JPEG images.
    
    Parameters:
    -----------
    video_path : str
        Path to the input AVI video file
    output_dir : str
        Directory where extracted frames will be saved
    max_frames : int, optional
        Maximum number of frames to extract (useful for testing)
    
    Returns:
    --------
    int
        Number of frames successfully extracted
    
    Raises:
    -------
    ValueError
        If video file cannot be opened or read
    """
    print(f"📹 Extracting frames from: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video file: {video_path}")
    
    # Get video properties for reporting
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Video properties: {total_frames} frames, {fps:.2f} FPS")
    
    frame_count = 0
    saved_count = 0
    
    # Read video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        # Optional: limit frames for testing
        if max_frames and saved_count >= max_frames:
            break
            
        # Save frame as JPEG with sequential naming
        output_path = os.path.join(output_dir, f"frame{saved_count}.jpg")
        # Use optimized JPEG compression (faster writing)
        success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            saved_count += 1
            frame_count += 1
        else:
            print(f"⚠️ Warning: Failed to save frame {saved_count}")
        
        # Progress reporting every 100 frames
        if frame_count % 100 == 0:
            print(f"📊 Extracted {frame_count} frames...")
    
    # Release video capture resource
    cap.release()
    
    if saved_count == 0:
        raise ValueError("❌ No frames were extracted from the video")
    
    print(f"✅ Successfully extracted {saved_count} frames to {output_dir}")
    return saved_count

def count_files(directory_path):
    """
    Count the number of files in a directory, ignoring hidden files.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory to scan
    
    Returns:
    --------
    int
        Number of non-hidden files in the directory
    """
    files = [f for f in os.listdir(directory_path) 
             if f.startswith('frame') and f.endswith('.jpg')]
    return len(files)

def preprocess_image_fast(image):
    """
    Apply preprocessing steps to enhance worm detection.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image
    
    Returns:
    --------
    numpy.ndarray
        Preprocessed image
    """
    processed = image.copy()
    
    # Apply Gaussian blur if enabled
    if GAUSSIAN_BLUR_SIZE > 0:
        processed = cv2.GaussianBlur(processed, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    
    # Apply binary threshold
    _, processed = cv2.threshold(processed, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    return processed

def interactive_worm_selection(frames_dir):
    """
    Interactive tool for manually selecting worm positions in the first frame.
    
    Parameters:
    -----------
    frames_dir : str
        Directory containing extracted frames
    
    Returns:
    --------
    numpy.ndarray
        Array of selected worm coordinates in format [[x1,y1], [x2,y2], ...]
    
    Raises:
    -------
    ValueError
        If first frame cannot be read or no worms are selected
    """
    print("🎯 Starting interactive worm selection...")
    first_frame_path = os.path.join(frames_dir, "frame0.jpg")
    frame = cv2.imread(first_frame_path)
    
    if frame is None:
        raise ValueError(f"❌ Could not read first frame: {first_frame_path}")
    
    # Create a copy for drawing without modifying original
    display_frame = frame.copy()
    coordinates = []
    
    def mouse_callback(event, x, y, flags, param):
        """
        Mouse callback function for OpenCV window.
        Handles left-click events to select worm positions.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append([x, y])
            worm_num = len(coordinates)
            print(f"🔴 Selected worm {worm_num}: [{x}, {y}]")
            
            # Visual feedback: draw circle and number at selection point
            cv2.circle(display_frame, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(display_frame, str(worm_num), (x+20, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("El Traquero - Worm Selection", display_frame)
    
    # Create interactive window
    window_name = "El Traquero - Worm Selection"
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # User instructions
    print("\n🎯 Interactive Worm Selection Instructions:")
    print("1. Click on the center of each worm in the image")
    print("2. Worms will be numbered in the order you click them")
    print("3. Press 'q' when finished selecting all worms")
    print("4. Press 'c' to clear all selections and start over")
    print("5. Press 'ESC' to cancel and exit")
    
    # Main interaction loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('c'):
            # Clear all selections and reset display
            coordinates.clear()
            display_frame = frame.copy()
            cv2.imshow(window_name, display_frame)
            print("🗑️ Selections cleared. Start over.")
    
    # PROPERLY CLOSE THE WINDOW
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Additional wait to ensure window closes
    
    # Validation
    if len(coordinates) == 0:
        raise ValueError("❌ No worms were selected! Please run again and select at least one worm.")
    
    # Summary of selections
    print(f"\n✅ Selected {len(coordinates)} worms:")
    for i, coord in enumerate(coordinates):
        print(f"   Worm {i+1}: [{coord[0]}, {coord[1]}]")
    
    return np.array(coordinates)

def apply_function_to_images_optimized(directory_path, Z, WORMSXY, DistLimit, Worms, OGWORMDATA):
    """
    Optimized main processing function that tracks worms across all frames.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing frame images
    Z : int
        Total number of frames to process
    WORMSXY : numpy.ndarray
        Pre-allocated array to store worm trajectories [frames × worms × 2]
    DistLimit : float
        Maximum distance for worm matching between frames (pixels)
    Worms : int
        Number of worms being tracked
    OGWORMDATA : numpy.ndarray
        Initial worm positions from first frame
    
    Returns:
    --------
    numpy.ndarray
        Complete worm trajectories across all frames
    """
    print(f"🔍 Starting to process {Z} frames for {Worms} worms...")
    print(f"📊 Tracking parameters: Max distance = {DistLimit}px, Min blob size = {MIN_BLOB_SIZE}px")
    
    if FRAME_SKIP > 1:
        print(f"⚡ Optimization: Processing every {FRAME_SKIP} frame(s) for speed")
    
    # Precompute squared distance limit to avoid sqrt in inner loop (performance optimization)
    squared_dist_limit = DistLimit * DistLimit
    
    for count in range(0, Z, FRAME_SKIP):
        # Progress reporting
        if count % 50 == 0:
            print(f"📊 Processing frame {count}/{Z}...")
            
        # Construct frame path and validate existence
        input_image_path = os.path.join(directory_path, f"frame{count}.jpg")
        
        if not os.path.exists(input_image_path):
            # Fill with previous positions for skipped/missing frames
            if count > 0:
                WORMSXY[count] = WORMSXY[count-1]
            continue
            
        # Read and validate frame (load directly as grayscale for performance)
        frame = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            if count > 0:
                WORMSXY[count] = WORMSXY[count-1]
            continue
            
        # Apply preprocessing
        image = preprocess_image_fast(frame)
        
        # Label connected components in the image
        TagParticles, NumParticles = label(image)
        
        # Filter out small particles (noise)
        image = ParticleSizeFilt(TagParticles, NumParticles, image, MIN_BLOB_SIZE)

        # Re-label after filtering
        TagParticles, NumParticles = label(image)
        featuresXY = np.empty([NumParticles, 2])

        # Calculate centroids of all remaining particles
        for l in range(NumParticles):
            centroidYX = center_of_mass(image, TagParticles, l + 1)
            centroid = np.array([centroidYX[1], centroidYX[0]])  # Convert to (x,y) format
            featuresXY[l] = centroid

        # Create backup of features for fallback matching
        fXYcache = featuresXY.copy()
        contador = 0

        # Match each worm to the closest particle in current frame
        for c3 in range(Worms):
            point = OGWORMDATA[c3]  # Current worm's previous position
            
            # Calculate distances to all particles in current frame
            if DISTANCE_CALCULATION == 'squared':
                # Use squared distances to avoid expensive sqrt operations
                differences = featuresXY - point
                squared_distances = np.sum(differences * differences, axis=1)
                closest_index = np.argmin(squared_distances)
                distance_check = squared_distances[closest_index] < squared_dist_limit
            else:
                # Use Euclidean distance (slower but more accurate)
                distances = np.linalg.norm(featuresXY - point, axis=1)
                closest_index = np.argmin(distances)
                distance_check = distances[closest_index] < DistLimit

            # Primary matching: use closest particle within distance limit
            if distance_check:
                WORMSXY[count, contador] = featuresXY[closest_index]
                # Mark this particle as "used" by setting to large values
                featuresXY[closest_index] = [9999999999, 9999999999]
            else:
                # Fallback matching: use cached features (before marking as used)
                if DISTANCE_CALCULATION == 'squared':
                    differences = fXYcache - point
                    squared_distances = np.sum(differences * differences, axis=1)
                    closest_index = np.argmin(squared_distances)
                    distance_check = squared_distances[closest_index] < squared_dist_limit
                else:
                    distances = np.linalg.norm(fXYcache - point, axis=1)
                    closest_index = np.argmin(distances)
                    distance_check = distances[closest_index] < DistLimit

                if distance_check:
                    WORMSXY[count, contador] = fXYcache[closest_index]
                else:
                    # Final fallback: use previous frame position
                    if count > 0:
                        WORMSXY[count, contador] = WORMSXY[count-1, contador]
                    else:
                        WORMSXY[count, contador] = point  # Use initial position for first frame

            contador += 1

        # Update OGWORMDATA with current positions for next frame
        for c4 in range(Worms):
            OGWORMDATA[c4][0] = WORMSXY[count][c4][0]
            OGWORMDATA[c4][1] = WORMSXY[count][c4][1]
            
        # Fill in skipped frames with current positions
        if FRAME_SKIP > 1:
            for skip_idx in range(1, FRAME_SKIP):
                next_frame = count + skip_idx
                if next_frame < Z:
                    WORMSXY[next_frame] = WORMSXY[count]

    print("✅ Frame processing completed")
    return WORMSXY

def plot_results_fast(WORMSXY, output_path=None):
    """
    Generate fast visualization plot of worm trajectories without interactive display.
    
    Parameters:
    -----------
    WORMSXY : numpy.ndarray
        Worm trajectory data in format [worms × coordinates × frames]
    output_path : str, optional
        Path to save the plot image. If None, plot is only displayed.
    """
    print("📊 Generating visualization plot...")
    
    # Create figure directly with desired size
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot each worm's trajectory
    for i in range(WORMSXY.shape[0]):
        ax.plot(WORMSXY[i][0], -1 * WORMSXY[i][1], 
                color=PLOT_COLORS[i % len(PLOT_COLORS)], 
                label=f'Worm {i+1}', 
                linewidth=PLOT_LINE_WIDTH,
                alpha=0.8)

    # Configure plot appearance
    ax.set_xticks(range(0, 1100, 100))
    ax.set_yticks(range(-1000, 100, 100))
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax.legend(loc='upper right')
    ax.set_title(f'El Traquero - Tracking Results ({WORMSXY.shape[0]} worms)', fontsize=16, pad=20)
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
        print(f"💾 Plot saved to: {output_path}")
    
    # Close the plot to free memory
    plt.close('all')
    print("📊 Plot generation completed")

def get_user_preferences():
    """
    Get user input for selecting worm identification method.
    
    Returns:
    --------
    numpy.ndarray or None
        Worm coordinates if manual method chosen, None for interactive selection
    """
    print("\n" + "="*60)
    print("🎯 El Traquero - Worm Tracking Setup")
    print("="*60)
    
    print(f"\n📊 Current Configuration:")
    print(f"   • Min blob size: {MIN_BLOB_SIZE} pixels")
    print(f"   • Max distance: {MAX_DISTANCE_LIMIT} pixels")
    print(f"   • Frame skip: {FRAME_SKIP}")
    print(f"   • Input video: {VIDEO_PATH}")
    
    print("\n🎯 How would you like to define worm positions?")
    print("1. 🖱️  Interactive selection (click on worms in first frame)")
    print("2. ⌨️  Manual coordinate input")
    
    choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip()
    
    if choice == "2":
        # Manual coordinate input method
        try:
            num_worms = int(input("How many worms are in the video? "))
            if num_worms <= 0:
                raise ValueError("Number of worms must be positive")
                
            coordinates = []
            print("\nEnter coordinates for each worm (format: x,y):")
            for i in range(num_worms):
                while True:
                    try:
                        coord_input = input(f"Worm {i+1} coordinates: ").strip()
                        x, y = map(int, coord_input.split(','))
                        coordinates.append([x, y])
                        break
                    except ValueError:
                        print("❌ Invalid format. Please use: x,y (e.g., 100,200)")
            
            print(f"\n✅ Manual input completed for {num_worms} worms")
            return np.array(coordinates)
            
        except ValueError as e:
            print(f"❌ Error in manual input: {e}")
            print("🔄 Switching to interactive selection...")
            return None
    else:
        # Default to interactive selection
        return None

def print_configuration_summary():
    """Print a comprehensive summary of the current configuration."""
    print("\n" + "="*60)
    print("⚙️  El Traquero - Configuration Summary")
    print("="*60)
    print(f"📁 Input/Output:")
    print(f"   • Video path: {VIDEO_PATH}")
    print(f"   • Frames directory: {FRAMES_DIR}")
    print(f"   • Output directory: {OUTPUT_DIR}")
    
    print(f"\n🎯 Tracking Parameters:")
    print(f"   • Min blob size: {MIN_BLOB_SIZE} pixels")
    print(f"   • Max distance limit: {MAX_DISTANCE_LIMIT} pixels")
    print(f"   • Extract all frames: {EXTRACT_ALL_FRAMES}")
    if not EXTRACT_ALL_FRAMES:
        print(f"   • Max frames to extract: {MAX_FRAMES_TO_EXTRACT}")
    
    print(f"\n⚡ Performance Optimizations:")
    print(f"   • Numba JIT: {NUMBA_AVAILABLE}")
    print(f"   • Frame skip: {FRAME_SKIP}")
    print(f"   • Distance calculation: {DISTANCE_CALCULATION}")
    print(f"   • Worm matching: {WORM_MATCHING_STRATEGY}")
    
    print(f"\n🖼️  Image Processing:")
    print(f"   • Gaussian blur: {GAUSSIAN_BLUR_SIZE} (0 = disabled)")
    print(f"   • Binary threshold: {BINARY_THRESHOLD}")
    print(f"   • Background subtraction: {BACKGROUND_SUBTRACTION}")
    
    print(f"\n📊 Visualization:")
    print(f"   • Plot colors: {len(PLOT_COLORS)} available")
    print(f"   • Background color: {PLOT_BACKGROUND_COLOR}")
    print(f"   • Line width: {PLOT_LINE_WIDTH}")
    print(f"   • Plot DPI: {PLOT_DPI}")
    print("="*60)

def main():
    """
    Main function coordinating the entire optimized worm tracking pipeline.
    
    Workflow:
    1. Extract frames from AVI video
    2. Select worm positions (interactive or manual)
    3. Track worms across all frames with optimizations
    4. Save results and generate visualization
    5. Report performance metrics
    """
    print("🚀 El Traquero - Starting OPTIMIZED Processing Pipeline")
    
    # Print configuration summary
    print_configuration_summary()
    
    # Create necessary directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 1: EXTRACT FRAMES FROM AVI VIDEO
    # =========================================================================
    if not os.path.exists(FRAMES_DIR) or count_files(FRAMES_DIR) == 0:
        print("🎬 Extracting frames from video...")
        max_frames = None if EXTRACT_ALL_FRAMES else MAX_FRAMES_TO_EXTRACT
        total_frames = extract_frames_from_video(VIDEO_PATH, FRAMES_DIR, max_frames)
    else:
        total_frames = count_files(FRAMES_DIR)
        print(f"📁 Using existing {total_frames} frames in {FRAMES_DIR}")
    
    # =========================================================================
    # STEP 2: WORM POSITION SELECTION
    # =========================================================================
    user_choice = get_user_preferences()
    
    if user_choice is not None:
        OGWORMDATA = user_choice
    else:
        print("\n🎯 Starting interactive worm selection...")
        OGWORMDATA = interactive_worm_selection(FRAMES_DIR)
    
    Worms = OGWORMDATA.shape[0]
    
    # =========================================================================
    # STEP 3: FRAME PROCESSING AND WORM TRACKING (OPTIMIZED)
    # =========================================================================
    Z = count_files(FRAMES_DIR)  # Total number of frames
    actual_frames_to_process = Z // FRAME_SKIP + (1 if Z % FRAME_SKIP else 0)
    print(f"\n🔍 Processing {actual_frames_to_process} of {Z} frames for {Worms} worms...")
    
    # Pre-allocate array for worm trajectories
    WORMSXY = np.empty([Z, Worms, 2])
    
    # Process all frames and track worms with optimizations
    WORMSXY = apply_function_to_images_optimized(FRAMES_DIR, Z, WORMSXY, MAX_DISTANCE_LIMIT, Worms, OGWORMDATA)
    
    # =========================================================================
    # STEP 4: SAVE RESULTS
    # =========================================================================
    # Transpose to format: [worms × coordinates × frames]
    WORMSXY_transposed = np.transpose(WORMSXY, (1, 2, 0))
    
    # Save individual CSV files for each worm
    for i in range(Worms):
        output_csv = os.path.join(OUTPUT_DIR, f"worm_{i+1}_trajectory.csv")
        savetxt(output_csv, WORMSXY_transposed[i], delimiter=",", header="x,y", comments='')
        print(f"💾 Saved worm {i+1} trajectory to: {output_csv}")
    
    # =========================================================================
    # STEP 5: VISUALIZATION (FAST)
    # =========================================================================
    plot_path = os.path.join(OUTPUT_DIR, "worm_trajectories.png")
    plot_results_fast(WORMSXY_transposed, plot_path)
    
    # Save configuration summary
    config_summary = f"""
    El Traquero - Processing Summary
    ================================
    Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}
    
    Input Configuration:
    - Video: {VIDEO_PATH}
    - Total frames: {Z}
    - Worms tracked: {Worms}
    
    Tracking Parameters:
    - Min blob size: {MIN_BLOB_SIZE} pixels
    - Max distance: {MAX_DISTANCE_LIMIT} pixels
    - Gaussian blur: {GAUSSIAN_BLUR_SIZE}
    - Binary threshold: {BINARY_THRESHOLD}
    
    Performance Optimizations:
    - Numba JIT: {NUMBA_AVAILABLE}
    - Frame skip: {FRAME_SKIP}
    - Distance calculation: {DISTANCE_CALCULATION}
    
    Results:
    - Trajectory files: {Worms} CSV files
    - Plot: worm_trajectories.png
    """
    
    with open(os.path.join(OUTPUT_DIR, "processing_summary.txt"), "w") as f:
        f.write(config_summary)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    print(f"\n" + "="*60)
    print("✅ EL TRAQUERO - PROCESSING COMPLETE")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   • Tracked {Worms} worms")
    print(f"   • Processed {Z} frames") 
    print(f"   • Generated {Worms} trajectory files")
    print(f"   • Created visualization plot")
    print(f"⏱️  Total processing time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"📈 Average: {total_time/Z:.3f} seconds per frame")
    print(f"💾 Results saved in: {OUTPUT_DIR}")
    print("="*60)
    
    return WORMSXY_transposed

if __name__ == "__main__":
    """
    Entry point of El Traquero script with comprehensive error handling.
    """
    print("El Traquero - Multi-Worm Tracking System")
    print("="*60)
    
    # Check for required packages
    required_packages = ['cv2', 'scipy', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'scipy':
                from scipy.ndimage import label, center_of_mass
            elif package == 'numpy':
                import numpy as np
            elif package == 'matplotlib':
                import matplotlib.pyplot as plt
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages. Please install with:")
        print(f"   pip install {' '.join(missing_packages)}")
        exit(1)
    
    # Check if input video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        print("💡 Please update the VIDEO_PATH variable in the script")
        exit(1)
    
    # =========================================================================
    # EXECUTION WITH ERROR HANDLING
    # =========================================================================
    try:
        # Execute main pipeline
        tracking_results = main()
        print("🎉 Processing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check that the VIDEO_PATH variable points to your AVI file")
        print("   2. Ensure the AVI file exists and is readable")
        print("   3. Verify file permissions")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 General troubleshooting:")
        print("   1. Ensure all packages are installed correctly")
        print("   2. Check that your AVI file is not corrupted")
        print("   3. Make sure you have a graphical interface for interactive selection")
        print("   4. Verify there is enough disk space for frame extraction")
        
    finally:
        print("\n🎯 El Traquero execution finished")
        # Force clean exit
        import sys
        sys.exit(0)
