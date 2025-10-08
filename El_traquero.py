#!/usr/bin/env python3
"""
El Traquero - Multi-Worm Tracking System
========================================

A "sophisticated system" for tracking multiple worms across video frames.
El Traquero ("Spanish" for “The Tracker”) automatically extracts frames from AVI videos, allows interactive worm selection, and tracks worm trajectories with customizable parameters. The system implements a centroid-based multi-object tracking algorithm, combining connected component analysis, blob filtering, and frame-to-frame nearest-neighbor matching to follow individual worms throughout long video sequences.
Each frame is preprocessed using adaptive thresholding and optional Gaussian blurring to enhance contrast and reduce noise. Worms are identified as distinct connected regions in the binary image, and their centroids are computed using scipy’s center_of_mass function. The algorithm then establishes correspondences between successive frames by minimizing spatial distances between centroids, constrained by a maximum displacement threshold. This approach provides an efficient and robust solution for slow-moving, non-overlapping organisms such as C. elegans.
El Traquero also integrates Numba JIT acceleration for real-time performance, and supports interactive initialization through a graphical interface that lets users manually select worm positions. The result is a versatile tracking pipeline that balances computational efficiency, accuracy, and usability — ideal for behavioral studies, locomotion analysis, or other biological imaging applications.

Authors: 
Jorge Alejandro Luna Herrera (a.k.a. Mr. Gusanos)
Natalia Soledad Lafourcade Luna Herrera (a.k.a. Nata, Natilla)
Sputnik Gregorio El Mar Luna Herrera (a.k.a. Goyo, Goyito)

Remastered by: 
Deepseek

Date: 08-oct-2025
Version: 3.0 - Rayo McQueen Edizione
"""

import os
import sys
import time
import math
import cv2
import numpy as np
from numpy import savetxt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try imports for accelerators
USE_PYCCEL = False
USE_NUMBA = False
PYCCEL_AVAILABLE = False
NUMBA_AVAILABLE = False

try:
    # Pyccel's high-level compile function
    from pyccel.epyccel import epyccel as pyccel_epyccel
    PYCCEL_AVAILABLE = True
    print("🔧 Pyccel available — will attempt to compile hot functions with Pyccel (preferred on Apple Silicon)")
except Exception:
    PYCCEL_AVAILABLE = False

try:
    from numba import njit, prange, config as numba_config
    NUMBA_AVAILABLE = True
    print("🔧 Numba available — will use Numba if Pyccel is not available")
except Exception:
    NUMBA_AVAILABLE = False

# =============================================================================
# USER CONFIGURATION SECTION - CUSTOMIZE THESE PARAMETERS FOR YOUR EXPERIMENT
# =============================================================================

# INPUT/OUTPUT PATHS (Update these for your system)
VIDEO_PATH = "/Users/jorgelunaherrera/Documents/Jorge - Tracking sample/output_video2.avi"          # Path to your input AVI video file
FRAMES_DIR = "/Users/jorgelunaherrera/Documents/Jorge - Tracking sample/extracted_frames5"         # Directory where frames will be extracted
OUTPUT_DIR = "/Users/jorgelunaherrera/Documents/Jorge - Tracking sample/extracted_frames5/results" # Directory for output files and results

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

# set OpenCV optimizations
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass

# Start timer
script_start_time = time.time()

# ----------------------------
# Helper: attempt to compile hot functions
# ----------------------------

def try_pyccel_compile(func, language='c', **epyccel_kwargs):
    """
    Try to compile `func` with Pyccel via epyccel.
    Returns the compiled function or None on failure.
    """
    if not PYCCEL_AVAILABLE:
        return None
    try:
        # epyccel returns a compiled callable
        compiled = pyccel_epyccel(func, language=language, **epyccel_kwargs)
        print(f"✅ Pyccel compiled function: {func.__name__} (language={language})")
        return compiled
    except Exception as e:
        print(f"⚠️ Pyccel compilation failed for {func.__name__}: {e}")
        return None

def try_numba_jit(func, nopython=True, parallel=False, fastmath=True):
    """
    Try to compile with Numba njit. Returns compiled or None on failure.
    """
    if not NUMBA_AVAILABLE:
        return None
    try:
        flags = {}
        # Use njit wrapper
        if parallel:
            compiled = njit(func, parallel=True, fastmath=fastmath)
        else:
            compiled = njit(func, fastmath=fastmath)
        # compile by calling once with dummy small arrays? we return compiled object
        print(f"✅ Numba-compiled function (njit) created for {func.__name__}")
        return compiled
    except Exception as e:
        print(f"⚠️ Numba compilation failed for {func.__name__}: {e}")
        return None

# ----------------------------
# HOT FUNCTION: particle_size_filter
# ----------------------------
# This function removes labels whose area < min_blob_size.
# We'll provide: (1) Pyccel compiled, (2) Numba compiled, (3) NumPy vectorized fallback.
def particle_size_filter_numpy(label_image, min_blob_size):
    """
    label_image: 2D int array (labels produced by connectedComponents)
    min_blob_size: int
    Returns: mask (boolean 2D) where True indicates KEEP pixels (not removed)
    """
    # labels are 0..N; compute area counts
    flat = label_image.ravel()
    if flat.size == 0:
        return np.zeros_like(label_image, dtype=bool)
    counts = np.bincount(flat)
    # ensure counts length at least 1
    # labels to keep: counts >= min_blob_size
    keep = counts >= min_blob_size
    # ensure background not kept
    if keep.shape[0] > 0:
        keep[0] = False
    # produce boolean mask: True where label in keep
    # np.isin is efficient in C
    mask = np.isin(label_image, np.nonzero(keep)[0])
    return mask

# Define a simple pure-Python version amenable to Pyccel/Numba compiling
def particle_size_filter_py(label_image, min_blob_size):
    """
    Pure Python implementation intended for compilation (Pyccel / Numba).
    Returns new label_image where removed labels are set to 0.
    Note: expects label_image as 2D array of ints.
    """
    # Build counts dictionary (simple approach)
    max_label = int(label_image.max()) if label_image.size > 0 else 0
    counts = [0] * (max_label + 1)
    rows, cols = label_image.shape
    for i in range(rows):
        for j in range(cols):
            v = int(label_image[i, j])
            counts[v] += 1
    # Determine which labels to remove
    remove = [False] * (max_label + 1)
    for lbl in range(len(counts)):
        if counts[lbl] < min_blob_size:
            remove[lbl] = True
    # Create mask: True = keep
    mask = np.empty((rows, cols), dtype=np.bool_)
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = not remove[int(label_image[i, j])]
    return mask

# Try compile particle_size_filter_py with Pyccel or Numba
particle_size_filter_compiled = None
if PYCCEL_AVAILABLE:
    particle_size_filter_compiled = try_pyccel_compile(particle_size_filter_py, language='c')
if particle_size_filter_compiled is None and NUMBA_AVAILABLE:
    # compile with njit - allow parallel? not necessary
    try:
        particle_size_filter_compiled = njit(particle_size_filter_py, nogil=True, cache=True)
        print("✅ particle_size_filter: Numba njit created")
    except Exception:
        particle_size_filter_compiled = None

# Use final selection function
def particle_size_filter(labels, min_blob_size):
    """
    Unified wrapper: returns boolean mask where True indicates KEEP pixels.
    Uses compiled version if available, otherwise NumPy vectorized fallback.
    """
    if particle_size_filter_compiled is not None:
        try:
            return particle_size_filter_compiled(labels, min_blob_size)
        except Exception as e:
            # Fall back if compiled version fails at runtime
            print("⚠️ Compiled particle_size_filter failed at runtime:", e)
    # NumPy fallback
    return particle_size_filter_numpy(labels, min_blob_size)

# ----------------------------
# HOT FUNCTION: match_worms (greedy assignment)
# ----------------------------
# Provide pure-Python function that Pyccel/Numba can compile, plus NumPy fallback.

def match_worms_py(features, prev_positions, squared_dist_limit):
    """
    features: (M,2) float array of candidate centroids (x,y)
    prev_positions: (W,2) float array of previous worm positions
    squared_dist_limit: float
    Returns matched_positions: (W,2) float array
    Pure Python loops - intended for compilation.
    """
    M = features.shape[0]
    W = prev_positions.shape[0]
    matched = np.empty((W, 2), dtype=np.float64)

    # If no features, just return previous positions
    if M == 0:
        for w in range(W):
            matched[w, 0] = prev_positions[w, 0]
            matched[w, 1] = prev_positions[w, 1]
        return matched

    # compute squared distances: M x W
    # naive nested loops to keep code simple for compilation
    dists = np.empty((M, W), dtype=np.float64)
    for m in range(M):
        fx = features[m, 0]
        fy = features[m, 1]
        for w in range(W):
            dx = fx - prev_positions[w, 0]
            dy = fy - prev_positions[w, 1]
            dists[m, w] = dx * dx + dy * dy

    used = np.zeros(M, dtype=np.bool_)
    assigned = -np.ones(W, dtype=np.int64)

    for w in range(W):
        best_idx = -1
        best_val = 1e18
        for m in range(M):
            if not used[m] and dists[m, w] < best_val:
                best_val = dists[m, w]
                best_idx = m
        if best_idx >= 0 and best_val < squared_dist_limit:
            assigned[w] = best_idx
            used[best_idx] = True

    for w in range(W):
        idx = assigned[w]
        if idx >= 0:
            matched[w, 0] = features[idx, 0]
            matched[w, 1] = features[idx, 1]
        else:
            # fallback: previous position
            matched[w, 0] = prev_positions[w, 0]
            matched[w, 1] = prev_positions[w, 1]

    return matched

def match_worms_numpy(features, prev_positions, squared_dist_limit):
    """
    Vectorized greedy matching fallback using NumPy.
    """
    W = prev_positions.shape[0]
    if features.shape[0] == 0:
        return prev_positions.copy()
    # distances: M x W
    diffs = features[:, None, :] - prev_positions[None, :, :]
    dists = np.sum(diffs * diffs, axis=2)
    assigned = -np.ones(W, dtype=np.int64)
    used = np.zeros(features.shape[0], dtype=bool)
    matched = np.empty_like(prev_positions)
    for w in range(W):
        col = dists[:, w].copy()
        col[used] = 1e18
        idx = np.argmin(col)
        if col[idx] < squared_dist_limit:
            assigned[w] = idx
            used[idx] = True
    for w in range(W):
        idx = assigned[w]
        if idx >= 0:
            matched[w] = features[idx]
        else:
            matched[w] = prev_positions[w]
    return matched

# Attempt to compile match_worms_py
match_worms_compiled = None
if PYCCEL_AVAILABLE:
    match_worms_compiled = try_pyccel_compile(match_worms_py, language='c')
if match_worms_compiled is None and NUMBA_AVAILABLE:
    try:
        match_worms_compiled = njit(match_worms_py, nogil=True, cache=True)
        print("✅ match_worms: Numba njit created")
    except Exception:
        match_worms_compiled = None

def match_worms(features, prev_positions, squared_dist_limit):
    """
    Unified wrapper for match_worms.
    Uses compiled version if available, else NumPy fallback.
    """
    if match_worms_compiled is not None:
        try:
            return match_worms_compiled(features, prev_positions, squared_dist_limit)
        except Exception as e:
            print("⚠️ Compiled match_worms failed at runtime:", e)
    return match_worms_numpy(features, prev_positions, squared_dist_limit)

# ----------------------------
# Image preprocessing utilities
# ----------------------------

def preprocess_image_fast(image):
    """
    Apply optional blur + thresholding.
    Maintains existing variable names (GAUSSIAN_BLUR_SIZE, BINARY_THRESHOLD).
    """
    img = image
    if GAUSSIAN_BLUR_SIZE and GAUSSIAN_BLUR_SIZE > 0:
        k = GAUSSIAN_BLUR_SIZE if GAUSSIAN_BLUR_SIZE % 2 == 1 else GAUSSIAN_BLUR_SIZE + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    # If BINARY_THRESHOLD set >0, use fixed threshold, else Otsu
    if BINARY_THRESHOLD is not None and BINARY_THRESHOLD > 0:
        _, binary = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # Return binary as uint8
    return binary.astype(np.uint8)

# ----------------------------
# Frame extraction (keeps your extract behavior)
# ----------------------------

def extract_frames_from_video(video_path, output_dir, max_frames=None):
    """
    Extract frames to output_dir using OpenCV (keeps your original behavior).
    """
    print(f"📹 Extracting frames from: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Video properties: {total_frames} frames, {fps:.2f} FPS")
    saved_count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and saved_count >= max_frames:
            break
        if frame_idx % FRAME_SKIP == 0:
            out_path = os.path.join(output_dir, f"frame{saved_count}.jpg")
            # Use high-quality JPEG for accuracy; lower if you want speed
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"📊 Extracted {frame_idx} frames so far...")
    cap.release()
    if saved_count == 0:
        raise ValueError("❌ No frames were extracted from the video")
    print(f"✅ Successfully extracted {saved_count} frames to {output_dir}")
    return saved_count

# ----------------------------
# Counting files
# ----------------------------
def count_files(directory_path):
    files = [f for f in os.listdir(directory_path) if f.startswith('frame') and f.endswith('.jpg')]
    return len(files)

# ----------------------------
# Main optimized processing function (keeps your signature)
# ----------------------------
def apply_function_to_images_optimized(directory_path, Z, WORMSXY, DistLimit, Worms, OGWORMDATA):
    """
    Optimized main processing function that tracks worms across all frames.
    Uses OpenCV connectedComponentsWithStats, compiled filters/matchers where available,
    and vectorized fallbacks.
    """
    print(f"🔍 Starting to process {Z} frames for {Worms} worms...")
    print(f"📊 Tracking parameters: Max distance = {DistLimit}px, Min blob size = {MIN_BLOB_SIZE}px")

    if FRAME_SKIP > 1:
        print(f"⚡ Optimization: Processing every {FRAME_SKIP} frame(s) for speed")

    # squared distance limit
    squared_limit = DistLimit * DistLimit

    # Precompute frame file paths
    frame_indices = list(range(0, Z, FRAME_SKIP))
    frame_paths = [os.path.join(directory_path, f"frame{idx}.jpg") for idx in frame_indices]

    # Preload frames to memory optionally (honor PRELOAD_FRAMES)
    preloaded = None
    if PRELOAD_FRAMES:
        print("📦 Preloading frames into memory (may use large RAM)...")
        preloaded = []
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 1))) as ex:
            futures = {ex.submit(cv2.imread, p, cv2.IMREAD_GRAYSCALE): p for p in frame_paths}
            for fut in as_completed(futures):
                p = futures[fut]
                img = fut.result()
                preloaded.append(img)
        # Note: preloaded indexed in same order as frame_paths

    # iterate through requested frames
    for loop_i, frame_idx in enumerate(frame_indices):
        count = frame_idx  # real frame number as in original
        if count % 50 == 0:
            print(f"📊 Processing frame {loop_i+1}/{len(frame_indices)} (frame {count})...")

        input_image_path = os.path.join(directory_path, f"frame{count}.jpg")
        if PRELOAD_FRAMES:
            # map frame_idx to index in preloaded list
            img = preloaded[loop_i] if loop_i < len(preloaded) else None
        else:
            if not os.path.exists(input_image_path):
                img = None
            else:
                img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # fill with previous
            if count > 0:
                WORMSXY[count] = WORMSXY[count - 1]
            continue

        image = preprocess_image_fast(img)

        # Connected components (fast OpenCV)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        if num_labels <= 1:
            # only background
            featuresXY = np.empty((0, 2), dtype=np.float64)
        else:
            # stats: rows correspond to labels 0..num_labels-1, area in CC_STAT_AREA
            areas = stats[:, cv2.CC_STAT_AREA]
            # mask out small blobs and background
            keep_labels_mask = areas >= MIN_BLOB_SIZE
            keep_labels_mask[0] = False  # background never kept
            kept_labels_indices = np.nonzero(keep_labels_mask)[0]
            if kept_labels_indices.size == 0:
                featuresXY = np.empty((0, 2), dtype=np.float64)
            else:
                # centroids are (x, y) floats
                features_all = centroids.astype(np.float64)
                featuresXY = features_all[kept_labels_indices]

        # Match worms
        # note: OGWORMDATA is in shape (Worms, 2) (x,y)
        matched_positions = match_worms(featuresXY, OGWORMDATA, squared_limit)

        if matched_positions is None:
            # fallback
            if count > 0:
                WORMSXY[count] = WORMSXY[count - 1]
            else:
                WORMSXY[count] = OGWORMDATA
        else:
            # store result
            WORMSXY[count] = matched_positions
            # update OGWORMDATA
            OGWORMDATA[:, 0] = matched_positions[:, 0]
            OGWORMDATA[:, 1] = matched_positions[:, 1]

        # fill skipped frames (if FRAME_SKIP > 1)
        if FRAME_SKIP > 1:
            for skip_idx in range(1, FRAME_SKIP):
                next_frame = count + skip_idx
                if next_frame < Z:
                    WORMSXY[next_frame] = WORMSXY[count]

    print("✅ Frame processing completed")
    return WORMSXY

# ----------------------------
# plotting and saving functions (kept simplified and fast)
# ----------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_results_fast(WORMSXY, output_path=None):
    """
    Generate visualization plot of worm trajectories without interactive display.
    Expects WORMSXY shape: [worms × coords × frames] or [worms,2,frames], but the caller
    sends transposed data in the same layout as original script.
    """
    print("📊 Generating visualization plot...")
    # If input is [worms,2,frames], convert to plotting friendly shapes
    # We'll expect WORMSXY shape [worms, 2, frames]
    if WORMSXY.ndim != 3:
        # attempt to transpose
        try:
            WORMSXY = np.transpose(WORMSXY, (0, 2, 1))
        except Exception:
            pass

    worms = WORMSXY.shape[0]
    frames = WORMSXY.shape[2]

    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(worms):
        xs = WORMSXY[i, 0, :]
        ys = WORMSXY[i, 1, :]
        ax.plot(xs, -1 * ys, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=PLOT_LINE_WIDTH, alpha=0.8)

    ax.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax.legend([f"Worm {i+1}" for i in range(worms)], loc='upper right')
    ax.set_title(f'El Traquero - Tracking Results ({worms} worms)', fontsize=16, pad=20)
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
        print(f"💾 Plot saved to: {output_path}")
    plt.close('all')
    print("📊 Plot generation completed")

# ----------------------------
# interactive worm selection (keeps original behavior)
# ----------------------------
def interactive_worm_selection(frames_dir):
    print("🎯 Starting interactive worm selection...")
    first_frame_path = os.path.join(frames_dir, "frame0.jpg")
    frame = cv2.imread(first_frame_path)
    if frame is None:
        raise ValueError(f"❌ Could not read first frame: {first_frame_path}")
    display_frame = frame.copy()
    coordinates = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append([x, y])
            worm_num = len(coordinates)
            cv2.circle(display_frame, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(display_frame, str(worm_num), (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("El Traquero - Worm Selection", display_frame)

    window_name = "El Traquero - Worm Selection"
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("\n🎯 Interactive Worm Selection Instructions:")
    print("1. Click on the center of each worm in the image")
    print("2. Worms will be numbered in the order you click them")
    print("3. Press 'q' when finished selecting all worms")
    print("4. Press 'c' to clear all selections and start over")
    print("5. Press 'ESC' to cancel and exit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            coordinates.clear()
            display_frame = frame.copy()
            cv2.imshow(window_name, display_frame)
            print("🗑️ Selections cleared. Start over.")
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if len(coordinates) == 0:
        raise ValueError("❌ No worms were selected! Please run again and select at least one worm.")
    for i, coord in enumerate(coordinates):
        print(f"   Worm {i+1}: [{coord[0]}, {coord[1]}]")
    return np.array(coordinates)

# ----------------------------
# Save results (CSV) similar to original
# ----------------------------
def save_results(WORMSXY_transposed, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    worms = WORMSXY_transposed.shape[0]
    for i in range(worms):
        output_csv = os.path.join(output_dir, f"worm_{i+1}_trajectory.csv")
        savetxt(output_csv, WORMSXY_transposed[i], delimiter=",", header="x,y", comments='')
        print(f"💾 Saved worm {i+1} trajectory to: {output_csv}")

# ----------------------------
# Main flow (keeps original structure and variable names)
# ----------------------------
def print_configuration_summary():
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
    print(f"   • Pyccel available: {PYCCEL_AVAILABLE}")
    print(f"   • Numba available: {NUMBA_AVAILABLE}")
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
    print("🚀 El Traquero - Starting OPTIMIZED Processing Pipeline")
    print_configuration_summary()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # STEP 1: EXTRACT FRAMES
    if not os.path.exists(FRAMES_DIR) or count_files(FRAMES_DIR) == 0:
        print("🎬 Extracting frames from video...")
        max_frames = None if EXTRACT_ALL_FRAMES else MAX_FRAMES_TO_EXTRACT
        total_frames = extract_frames_from_video(VIDEO_PATH, FRAMES_DIR, max_frames)
    else:
        total_frames = count_files(FRAMES_DIR)
        print(f"📁 Using existing {total_frames} frames in {FRAMES_DIR}")

    # STEP 2: SELECT WORM POSITIONS
    # Allow manual override via environment variable or existing coordinates (keeps original flow)
    user_choice = None
    try:
        user_choice = None  # keep original get_user_preferences if you want; default to interactive
    except Exception:
        user_choice = None

    if user_choice is not None:
        OGWORMDATA = user_choice
    else:
        OGWORMDATA = interactive_worm_selection(FRAMES_DIR)

    Worms = OGWORMDATA.shape[0]

    # STEP 3: PROCESS FRAMES AND TRACK
    Z = count_files(FRAMES_DIR)
    # Make WORMSXY in original shape: [Z, Worms, 2]
    WORMSXY = np.empty([Z, Worms, 2], dtype=np.float64)

    WORMSXY = apply_function_to_images_optimized(FRAMES_DIR, Z, WORMSXY, MAX_DISTANCE_LIMIT, Worms, OGWORMDATA.copy())

    # STEP 4: SAVE RESULTS (transpose to [worms, 2, frames] to match plotting)
    WORMSXY_transposed = np.transpose(WORMSXY, (1, 2, 0))
    save_results(WORMSXY_transposed, OUTPUT_DIR)

    # STEP 5: PLOT
    plot_path = os.path.join(OUTPUT_DIR, "worm_trajectories.png")
    plot_results_fast(WORMSXY_transposed, plot_path)

    # config summary file
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
    - Pyccel available: {PYCCEL_AVAILABLE}
    - Numba available: {NUMBA_AVAILABLE}
    - Frame skip: {FRAME_SKIP}
    - Distance calculation: {DISTANCE_CALCULATION}

    Results:
    - Trajectory files: {Worms} CSV files
    - Plot: worm_trajectories.png
    """
    with open(os.path.join(OUTPUT_DIR, "processing_summary.txt"), "w") as f:
        f.write(config_summary)

    total_time = time.time() - script_start_time
    print("\n" + "="*60)
    print("✅ EL TRAQUERO - PROCESSING COMPLETE")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   • Tracked {Worms} worms")
    print(f"   • Processed {Z} frames")
    print(f"   • Generated {Worms} trajectory files")
    print(f"   • Created visualization plot")
    print(f"⏱️  Total processing time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"📈 Average: {total_time/Z if Z>0 else 0:.3f} seconds per frame")
    print(f"💾 Results saved in: {OUTPUT_DIR}")
    print("="*60)

    return np.transpose(WORMSXY, (1, 2, 0))

# ----------------------------
# Entry point with checks
# ----------------------------
if __name__ == "__main__":
    print("El Traquero - Multi-Worm Tracking System")
    print("="*60)

    # Quick dependency check
    required_packages = ['cv2', 'numpy', 'matplotlib']
    missing = []
    try:
        import cv2  # noqa
    except Exception:
        missing.append('opencv-python (cv2)')
    try:
        import numpy  # noqa
    except Exception:
        missing.append('numpy')
    try:
        import matplotlib  # noqa
    except Exception:
        missing.append('matplotlib')

    if missing:
        print("❌ Missing required packages:", missing)
        print("Install with: pip install opencv-python numpy matplotlib")
        sys.exit(1)

    if not os.path.exists(VIDEO_PATH) and not os.path.exists(FRAMES_DIR):
        print(f"❌ Neither VIDEO_PATH ({VIDEO_PATH}) nor FRAMES_DIR ({FRAMES_DIR}) exist.")
        print("Please update the USER CONFIGURATION SECTION at the top of the script.")
        sys.exit(1)

    try:
        results = main()
        print("🎉 Processing completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print("❌ Unexpected error:", e)
        import traceback
        traceback.print_exc()
    finally:
        print("\n🎯 El Traquero execution finished")
        sys.exit(0)
