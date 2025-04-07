import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import time
import logging
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("image_processing.log")
    ]
)

def get_image_resolution(image_path):
    """Get the resolution of an image without loading the full image."""
    try:
        # This is faster than reading the whole image
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)  # Load at reduced size for speed
        if img is None:
            return None
        # Get actual dimensions (not the reduced ones)
        img_full = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_full is None:
            return None
        return img_full.shape[1], img_full.shape[0]  # width, height
    except Exception as e:
        logging.error(f"Error reading {image_path}: {e}")
        return None

def get_reference_resolution(input_dir, sample_size=100):
    """
    Get the most common resolution from images in the input directory.
    Uses sampling for speed with large datasets.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    all_image_files = []
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            all_image_files.append(os.path.join(input_dir, filename))
    
    # Sample a subset of images for faster processing
    if len(all_image_files) > sample_size:
        import random
        sampled_files = random.sample(all_image_files, sample_size)
    else:
        sampled_files = all_image_files
    
    logging.info(f"Checking resolutions of {len(sampled_files)} images (from {len(all_image_files)} total)")
    
    # Get resolutions using multiple processes
    resolutions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
        futures = {executor.submit(get_image_resolution, img_path): img_path for img_path in sampled_files}
        for future in concurrent.futures.as_completed(futures):
            resolution = future.result()
            if resolution:
                resolutions.append(resolution)
    
    # Find the most common resolution
    if not resolutions:
        raise ValueError(f"No valid images found in {input_dir}")
    
    # Count occurrences of each resolution
    resolution_counts = {}
    for res in resolutions:
        resolution_counts[res] = resolution_counts.get(res, 0) + 1
    
    # Find the most common resolution
    most_common_resolution = max(resolution_counts.items(), key=lambda x: x[1])[0]
    logging.info(f"Reference resolution: {most_common_resolution[0]}x{most_common_resolution[1]}")
    
    return most_common_resolution

def process_image(img_path, target_width, target_height, output_dir, method='resize'):
    """Process a single image - suitable for parallel processing"""
    try:
        # Skip if output already exists
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        if os.path.exists(output_path):
            return output_path, True
            
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Could not read image: {img_path}")
            return None, False
        
        # Get current resolution
        current_height, current_width = img.shape[:2]
        
        # Skip if already matching target resolution
        if current_width == target_width and current_height == target_height:
            cv2.imwrite(output_path, img)
            return output_path, True
            
        # Standardize the image according to the chosen method
        if method == 'resize':
            # Simple resize (may distort aspect ratio)
            standardized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        elif method == 'pad':
            # Resize while maintaining aspect ratio, then pad
            aspect_ratio = min(target_width / current_width, target_height / current_height)
            new_width = int(current_width * aspect_ratio)
            new_height = int(current_height * aspect_ratio)
            
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create a black canvas with target dimensions
            standardized = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate offsets to center the image in the canvas
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Place the resized image on the canvas
            standardized[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        elif method == 'crop':
            # Resize to cover the target dimensions, then center crop
            aspect_ratio = max(target_width / current_width, target_height / current_height)
            new_width = int(current_width * aspect_ratio)
            new_height = int(current_height * aspect_ratio)
            
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Calculate crop coordinates
            x_offset = (new_width - target_width) // 2
            y_offset = (new_height - target_height) // 2
            
            # Crop to target dimensions
            standardized = resized[y_offset:y_offset + target_height, x_offset:x_offset + target_width]
        
        # Save the standardized image
        cv2.imwrite(output_path, standardized)
        return output_path, True
    
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None, False

def match_csv_image_resolution(csv_path, input_dir, output_dir, image=None, method='resize', 
                               max_workers=None, batch_size=100):
    """
    Match the resolution of images referenced in the CSV file to the most common
    resolution in the input directory.
    
    Args:
        csv_path (str): Path to the CSV file containing image references
        input_dir (str): Directory containing reference images for target resolution
        output_dir (str): Directory to save standardized images
        image (str): Name of the column in CSV containing image paths/names
        method (str): Resizing method ('resize', 'pad', or 'crop')
        max_workers (int): Maximum number of worker threads (None = auto)
        batch_size (int): How many images to process in each batch
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting CSV image resolution matching")
    logging.info(f"CSV: {csv_path}")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Get reference resolution from input directory
    target_width, target_height = get_reference_resolution(input_dir)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    logging.info(f"CSV contains {len(df)} rows")
    
    # Determine which column contains image information
    if image is None:
        # Try to automatically detect the image column
        possible_image_cols = ['image', 'imagepath', 'image_path', 'filename', 'file']
        for col in possible_image_cols:
            if col in df.columns or col.title() in df.columns or col.upper() in df.columns:
                if col in df.columns:
                    image = col
                elif col.title() in df.columns:
                    image = col.title()
                elif col.upper() in df.columns:
                    image = col.upper()
                break
        
        if image is None:
            # If still not found, use the first column that seems to contain image paths
            for col in df.columns:
                if len(df) > 0:
                    sample_val = str(df[col].iloc[0])
                    if any(sample_val.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']):
                        image = col
                        break
    
    if image is None or image not in df.columns:
        raise ValueError(f"Could not determine image column in CSV. Available columns: {', '.join(df.columns)}")
    
    logging.info(f"Using column '{image}' for image paths")
    
    # Process each image referenced in the CSV
    processed_count = 0
    failed_count = 0
    csv_base_dir = os.path.dirname(csv_path)
    
    # Get all image paths from CSV
    img_paths = []
    full_img_paths = []
    
    for index, row in df.iterrows():
        # Get image path from CSV
        img_path = row[image]
        
        # Handle relative paths in CSV
        if not os.path.isabs(img_path):
            # First try relative to CSV file
            full_img_path = os.path.join(csv_base_dir, img_path)
            if not os.path.exists(full_img_path):
                # Try common image folders
                for img_dir in ['images', 'img', 'data']:
                    potential_path = os.path.join(csv_base_dir, img_dir, img_path)
                    if os.path.exists(potential_path):
                        full_img_path = potential_path
                        break
                        
                if not os.path.exists(full_img_path):
                    # As a last resort, try finding the file by name in the input directory
                    img_name = os.path.basename(img_path)
                    potential_path = os.path.join(input_dir, img_name)
                    if os.path.exists(potential_path):
                        full_img_path = potential_path
        else:
            full_img_path = img_path
        
        # Check if the image exists
        if not os.path.exists(full_img_path):
            logging.warning(f"Image not found: {img_path} (tried {full_img_path})")
            continue
        
        img_paths.append(img_path)
        full_img_paths.append(full_img_path)
    
    logging.info(f"Found {len(full_img_paths)} valid images to process")
    
    # Process images in batches to avoid memory issues
    new_img_paths = []
    
    for i in range(0, len(full_img_paths), batch_size):
        batch_img_paths = full_img_paths[i:i+batch_size]
        batch_original_paths = img_paths[i:i+batch_size]
        
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(full_img_paths) + batch_size - 1)//batch_size}")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_image, 
                    img_path, 
                    target_width, 
                    target_height, 
                    output_dir, 
                    method
                ): (idx, orig_path) 
                for idx, (img_path, orig_path) in enumerate(zip(batch_img_paths, batch_original_paths))
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(batch_img_paths),
                desc=f"Batch {i//batch_size + 1}"
            ):
                idx, orig_path = futures[future]
                result_path, success = future.result()
                
                if success:
                    new_img_paths.append((idx + i, orig_path, result_path))
                    processed_count += 1
                else:
                    failed_count += 1
                    new_img_paths.append((idx + i, orig_path, None))
    
    # Create an updated CSV with the new image paths
    updated_df = df.copy()
    
    # Sort results by original index
    new_img_paths.sort(key=lambda x: x[0])
    
    # Update image paths in the DataFrame
    for idx, orig_path, new_path in new_img_paths:
        if new_path:
            # Use relative path from output directory
            relative_path = os.path.basename(new_path)
            updated_df.at[idx, image] = relative_path
    
    # Save updated CSV
    updated_csv_path = os.path.join(output_dir, os.path.basename(csv_path))
    updated_df.to_csv(updated_csv_path, index=False)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Finished processing {processed_count} images ({failed_count} failed)")
    logging.info(f"Standardized to resolution {target_width}x{target_height}")
    logging.info(f"Total time: {elapsed_time:.2f} seconds")
    logging.info(f"Updated CSV saved to: {updated_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Match CSV image resolution to reference images')
    parser.add_argument('--csv', required=True, help='Path to CSV file containing image references')
    parser.add_argument('--input-dir', required=True, help='Directory with reference images for target resolution')
    parser.add_argument('--output-dir', required=True, help='Directory to save standardized images')
    parser.add_argument('--image', help='Column name in CSV containing image paths')
    parser.add_argument('--method', default='resize', choices=['resize', 'pad', 'crop'], 
                        help='Method for standardizing resolution')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of worker threads (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    
    args = parser.parse_args()
    
    match_csv_image_resolution(
        args.csv, 
        args.input_dir, 
        args.output_dir, 
        args.image,
        args.method,
        args.workers,
        args.batch_size
    )

if __name__ == "__main__":
    main()