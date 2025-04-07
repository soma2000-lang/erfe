import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def standardize_resolution(input_dir, output_dir, target_width, target_height, keep_aspect_ratio=False, padding=False):
    """
    Standardize the resolution of all images in the input directory.
    
    Args:
        input_dir (str): Directory containing images to be processed
        output_dir (str): Directory where processed images will be saved
        target_width (int): Target width for all images
        target_height (int): Target height for all images
        keep_aspect_ratio (bool): Whether to maintain aspect ratio when resizing
        padding (bool): Whether to add padding after resizing to reach target dimensions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        # Read the image
        img_path = os.path.join(input_dir, image_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {img_path}, skipping...")
                continue
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue
        
        if keep_aspect_ratio:
            if padding:
                # Resize the image while maintaining aspect ratio
                h, w = img.shape[:2]
                ratio = min(target_width / w, target_height / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create a black canvas with target dimensions
                standardized = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                
                # Calculate offsets to center the image in the canvas
                x_offset = (target_width - new_w) // 2
                y_offset = (target_height - new_h) // 2
                
                # Place the resized image on the canvas
                standardized[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            else:
                # Resize the image while maintaining aspect ratio, without padding
                h, w = img.shape[:2]
                ratio = min(target_width / w, target_height / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                standardized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Directly resize to target dimensions (may distort aspect ratio)
            standardized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Save the standardized image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, standardized)
    
    print(f"Finished processing {len(image_files)} images. Output saved to {output_dir}")
    
def main():
    parser = argparse.ArgumentParser(description='Standardize image resolution')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for standardized images')
    parser.add_argument('--width', type=int, required=True, help='Target width')
    parser.add_argument('--height', type=int, required=True, help='Target height')
    parser.add_argument('--keep-aspect', action='store_true', help='Keep aspect ratio')
    parser.add_argument('--padding', action='store_true', help='Add padding to maintain target dimensions')
    
    args = parser.parse_args()
    
    standardize_resolution(
        args.input, 
        args.output, 
        args.width, 
        args.height, 
        args.keep_aspect, 
        args.padding
    )

if __name__ == "__main__":
    main()