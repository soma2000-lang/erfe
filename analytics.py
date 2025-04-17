import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from collections import defaultdict


def load_annotations(json_path):
    """Load annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def is_night_image(img_path, threshold=100):
   
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image at {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold

def calculate_size_ratio(data, img_dir):
  
    results = []

    images = data.get('images', [])
    annotations = data.get('annotations', [])

    image_details = {}
    for img in images:
        image_details[img['id']] = {
            'filename': img['file_name'],
            'height': img['height'],
            'width': img['width']
        }

    for annotation in annotations:
        image_id = annotation['image_id']
        img_info = image_details.get(image_id)
        
        if not img_info:
            print(f"Warning: No image info found for id {image_id}")
            continue
        
        filename = img_info['filename']
        img_height = img_info['height']
        img_width = img_info['width']
        img_area = img_height * img_width
        img_path = os.path.join(img_dir, filename)
        if 'bbox' in annotation:
            bbox = annotation['bbox']
            if len(bbox) >= 4:
                obj_width = bbox[2]  
                obj_height = bbox[3]  
                obj_area = obj_width * obj_height
            else:
                print(f"Warning: Unexpected bbox format for annotation with id {annotation.get('id')}")
                continue
        else:
            print(f"Warning: No area or bbox found for annotation with id {annotation.get('id')}")
            continue

        ratio = obj_area / img_area
        is_night = False
        if os.path.exists(img_path):
            is_night = is_night_image(img_path)
        
        
        results.append({
            'filename': filename,
            'image_id': image_id,
            'annotation_id': annotation.get('id'),
            'object_area': obj_area,
            'image_area': img_area,
            'image_height': img_height,
            'image_width': img_width,
            'ratio': ratio,
            'is_night': is_night
        })
    
    return results

def analyze_results(results):
    all_ratios = [r['ratio'] for r in results]
    stats = {
        'count': len(all_ratios),
        'avg_ratio': np.mean(all_ratios) if all_ratios else 0,
        'std_ratio': np.std(all_ratios) if all_ratios else 0
      
    }
    
    # Night images analysis
    night_results = [r for r in results if r['is_night']]
    night_ratios = [r['ratio'] for r in night_results]
    night_stats = {
        'count': len(night_ratios),
        'avg_ratio': np.mean(night_ratios) if night_ratios else 0,
        'std_ratio': np.std(all_ratios) if all_ratios else 0

    }
    
    return {
        'all_images': stats,
        'night_images': night_stats,
        'results': results
    }

def print_summary(analysis):
   
    all_stats = analysis['all_images']
    night_stats = analysis['night_images']
    

    print(f"\nTotal images count: {all_stats['count']}")
    print(f"Night images: {night_stats['count']} ({night_stats['count']/all_stats['count']*100:.1f}% of total)")
    print(f"Average ratio of all images: {all_stats['avg_ratio']:.6f}")
    print(f"Average ratio of night images: {night_stats['avg_ratio']:.6f}")
    print(f"Standard deviation of all images: {all_stats['std_ratio']:.6f}")
    print(f"Standard deviation of night images: {night_stats['std_ratio']:.6f}")


def main():


    json_path = '/home/AD/smajumder/lard_nominal/LARDS_test/synthetic_test/annotations.json'  
    img_dir = '/home/AD/smajumder/lard_nominal/LARDS_test/synthetic_test/images'  
    
    try:
        data = load_annotations(json_path)
        results = calculate_size_ratio(data, img_dir)
        analysis = analyze_results(results)
        print_summary(analysis)
    except Exception as e:

        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()