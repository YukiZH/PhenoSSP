#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Data Preprocessing (Patch Extraction)
Goal: Extract single-cell patches from raw mIF images based on segmentation masks.

Usage:
    python 01_data_preprocessing.py --input_dir /path/to/raw_dataset --output_dir /path/to/save_patches
"""

import os
import glob
import tifffile
import numpy as np
from skimage import measure
from tqdm import tqdm
import warnings
import zarr
import argparse
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Global variables for worker processes
global_config = {}

def init_worker(input_dir, output_dir, patch_size):
    """Initialize global variables in worker processes."""
    global_config['BASE_DATA_DIR'] = input_dir
    global_config['OUTPUT_PATCH_DIR'] = output_dir
    global_config['PATCH_SIZE'] = patch_size

def process_sample(mask_path):
    """
    Process a single sample: extract patches based on the mask.
    This function runs in parallel worker processes.
    """
    try:
        # Retrieve configuration from global state
        base_dir = global_config['BASE_DATA_DIR']
        output_root = global_config['OUTPUT_PATCH_DIR']
        patch_size = global_config['PATCH_SIZE']
        
        # Infer cohort_id and sample_id from file structure
        # Assumes structure: base_dir/cohort_id/mask/sample_id_mask.tif
        cohort_id = os.path.basename(os.path.dirname(os.path.dirname(mask_path)))
        sample_id = os.path.basename(mask_path).replace('_mask.tif', '')
        
        raw_dir = os.path.join(base_dir, cohort_id, 'raw')
        
        # Search for corresponding raw image
        image_path_search = glob.glob(os.path.join(raw_dir, f"*_{sample_id}.ome.tiff"))
        if not image_path_search:
            # Try searching without "ome" prefix if strictly matching logic failed
            image_path_search = glob.glob(os.path.join(raw_dir, f"*{sample_id}*.tiff"))
            
        if not image_path_search:
            return 0
        
        image_path = image_path_search[0]
        
        # Optimization: Use zarr for memory-mapped reading (critical for large TIFFs)
        image_zarr_store = tifffile.imread(image_path, aszarr=True)
        image = zarr.open(image_zarr_store, mode='r')
        
        mask = tifffile.imread(mask_path)
        
        # Check channel dimension (Channels-Last vs Channels-First)
        is_transposed = image.shape[-1] == 7
        
        props = measure.regionprops(mask)
        
        output_sample_dir = os.path.join(output_root, cohort_id, sample_id)
        os.makedirs(output_sample_dir, exist_ok=True)
        
        half_size = patch_size // 2
        cells_processed = 0

        for prop in props:
            if prop.area < 20: continue # Skip artifacts
            
            cy, cx = prop.centroid
            cy, cx = int(cy), int(cx)

            # Slicing logic based on image layout
            if is_transposed: # (H, W, C)
                patch = image[max(0, cy - half_size):cy + half_size, max(0, cx - half_size):cx + half_size, :]
                padded_patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
                h, w = patch.shape[0], patch.shape[1]
                padded_patch[:h, :w, :] = patch
                # Convert to (C, H, W) for PyTorch
                padded_patch = np.transpose(padded_patch, (2, 0, 1)) 
            else: # (C, H, W)
                patch = image[:, max(0, cy - half_size):cy + half_size, max(0, cx - half_size):cx + half_size]
                padded_patch = np.zeros((image.shape[0], patch_size, patch_size), dtype=image.dtype)
                h, w = patch.shape[1], patch.shape[2]
                padded_patch[:, :h, :w] = patch

            cell_id = prop.label
            save_path = os.path.join(output_sample_dir, f"cell_{cell_id}.npy")
            np.save(save_path, padded_patch)
            cells_processed += 1
            
        return cells_processed
    
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return 0

def parse_args():
    parser = argparse.ArgumentParser(description="PhenoSSP Data Preprocessing")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Root directory of the dataset (containing cohort folders)')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_patches', 
                        help='Directory to save extracted patches')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of single-cell patches (default: 64)')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2), 
                        help='Number of parallel worker processes')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("--- [Step 1] Patch Extraction & Preprocessing ---")
    print(f"📂 Input: {args.input_dir}")
    print(f"📂 Output: {args.output_dir}")
    print(f"🚀 Workers: {args.workers}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Collect all mask files
    print("--> Scanning for mask files...")
    all_mask_files = []
    
    # Assumes directory structure: input_dir / cohort_id / mask / *.tif
    # Adjust this glob pattern if your raw data structure is different
    cohort_dirs = [d for d in glob.glob(os.path.join(args.input_dir, '*')) if os.path.isdir(d)]
    
    for cohort_dir in cohort_dirs:
        mask_dir = os.path.join(cohort_dir, 'mask')
        if os.path.exists(mask_dir):
            masks = glob.glob(os.path.join(mask_dir, '*.tif'))
            all_mask_files.extend(masks)
    
    if not all_mask_files:
        print("❌ No mask files found! Please check the input directory structure.")
        print(f"Expected structure: {args.input_dir}/<cohort_id>/mask/*.tif")
        return

    print(f"--> Found {len(all_mask_files)} samples to process.")

    # 2. Parallel Processing
    total_cells_extracted = 0
    
    # Pass arguments to worker processes
    with Pool(processes=args.workers, initializer=init_worker, initargs=(args.input_dir, args.output_dir, args.patch_size)) as pool:
        with tqdm(total=len(all_mask_files), desc="Extracting Patches") as pbar:
            for count in pool.imap_unordered(process_sample, all_mask_files):
                total_cells_extracted += count
                pbar.update(1)
                if total_cells_extracted > 0:
                    pbar.set_postfix_str(f"Cells: {total_cells_extracted}")
                
    print(f"\n🎉 Preprocessing Complete!")
    print(f"✅ Total cells extracted: {total_cells_extracted}")
    print(f"✅ Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()