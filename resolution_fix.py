import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

def analyze_image_directory(directory_path):
    image_data = []
    extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    
    for root, _, files in os.walk(directory_path):
        for file in tqdm(files, desc="Analyzing images"):

            if any(file.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        image_data.append({
                            'filename': file,
                            'path': file_path,
                            'width': width,
                            'height': height,
                            'resolution': f"{width}x{height}"
                        })
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
   
    df = pd.DataFrame(image_data)

    resolution_counts = df['resolution'].value_counts()
    print(f"Found {len(df)} images with {len(resolution_counts)} different resolutions")
    print("\nResolution distribution:")
    print(resolution_counts)
    
    return df

def resize_images(df, target_size=(512, 512), output_dir=None):
    if output_dir is None:
        output_dir = "resized_images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resizing images"):
        try:
            with Image.open(row['path']) as img:
               
                img_resized = img.resize(target_size, Image.LANCZOS)
                
            
                output_path = os.path.join(output_dir, os.path.basename(row['path']))
                img_resized.save(output_path)
        except Exception as e:
            print(f"Error resizing {row['path']}: {e}")
    
    print(f"All images resized to {target_size[0]}x{target_size[1]} and saved to {output_dir}")

if __name__ == "__main__":
    image_dir = "/home/AD/smajumder/lard/data/"  
    df = analyze_image_directory(image_dir)

    target_size = (2648, 2448)  
  
    response = input(f"Resize all images to {target_size[0]}x{target_size[1]}? (y/n): ")
    if response.lower() == 'y':
        resize_images(df, target_size=target_size)