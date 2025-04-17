import cv2
import numpy as np
import json
import time
import os
import onnxruntime
import albumentations as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import torch
from scipy import ndimage
from tqdm import tqdm
import onnx
from glob import glob 

def get_onnx_session(onnx_model_path, use_gpu=True):  # gpu might gg
    """Create an optimized ONNX runtime session with GPU support if available"""
    # Check if GPU is available
    providers = []
    # print(onnxruntime.get_available_providers())
    if use_gpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers.append('CUDAExecutionProvider')
        # print("Using CUDA for inference")
    else:
        if use_gpu:
            print("CUDA requested but not available. Falling back to CPU.")
        providers.append('CPUExecutionProvider')
        # print("Using CPU for inference")
    
    # Set session options for better performance
    session_options = onnxruntime.SessionOptions()
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create session with configured providers
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        sess_options=session_options,
        providers=providers
    )
    
    return session

def preprocess_image(image, max_size=1024):
    """Preprocess image according to the model requirements"""
    aug_transforms = []
    rescale_transforms = []
    rescale_transforms.append(A.LongestMaxSize(max_size, p=1.0))
    rescale_transforms.append(
            A.PadIfNeeded(
                max_size,
                max_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
                p=1.0,
            )
        )
    aug_transforms.append(A.Sequential(rescale_transforms, p=1.0))
    aug_transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    transform = A.Compose(aug_transforms)
    transformed = transform(image=image)
    
   
    
    return transformed["image"]
def batch_preprocess_images(images, max_size=1024):
    """Preprocess a batch of images"""
    processed_images = []
  
    
    for image in images:
        processed_img = preprocess_image(image, max_size)
        processed_images.append(processed_img)
       
    # Stack images into a batch
    batch_images = np.stack(processed_images, axis=0)
    
    return batch_images


    

def run_batch_inference(session, batch_images):
    """Run inference on a batch of images"""
    input_name = session.get_inputs()[0].name
    
    # Get all output names to see what the model is actually returning
    output_names = [output.name for output in session.get_outputs()]
    # print(f"Model output names: {output_names}")
    
    # Prepare input
    input_data = batch_images.transpose(0, 3, 1, 2).astype(np.float32)
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = time.time() - start_time
    
    # Debug all outputs
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}, min: {output.min()}, max: {output.max()}")
    
    return outputs[0], inference_time

# mask logits se mask
def postprocess_masks(mask_logits, threshold=0.49, apply_sigmoid: bool = False):
    """Apply sigmoid and threshold to get binary masks for a batch"""
    if apply_sigmoid:
        sigmoid_masks = 1 / (1 + np.exp(-mask_logits))
    else:
        sigmoid_masks = mask_logits
        
   
    print("sigmoid masks",sigmoid_masks)
    # Try using channel 1 instead of channel 0
    binary_masks = (sigmoid_masks[:, 1] > threshold).astype(np.uint8)
    
    print("binary masks should get [0,1]",binary_masks) # s
    
    return binary_masks  
def extract_contours(mask, min_area=0): # getting contours from the binary masks
    """Extract contours from the mask"""
 
    mask_uint8 = mask.astype(np.uint8)# OpenCV compatibility: Many
    
    # # print mask statistics for debugging
    # print(f"Mask shape: {mask_uint8.shape}, dtype: {mask_uint8.dtype}")
    # print(f"Mask values: min={mask_uint8.min()}, max={mask_uint8.max()}, sum={np.sum(mask_uint8)}")
    
    # If the mask is empty, return empty contours
    if np.sum(mask_uint8) == 0:
        # print("Warning: Empty mask!")
        return []
        
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    
    return filtered_contours
def get_quadrilateral( filtered_contours): # we should get the mask as in the gt, coordianes wale plotting type
    """Extract a quadrilateral from the contour"""
    # Get minimum area rectangle
    rect = cv2.minAreaRect(filtered_contours)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def get_bbox(filtered_contour): # should cover the runaway
    """Get bounding box from contour"""
    x, y, w, h = cv2.boundingRect(filtered_contour)
    return [x, y, x + w, y + h]  # [x1, y1, x2, y2] format

def calculate_model_params(onnx_model):
    """Calculate and return number of parameters in the model"""
    total_params = 0
    
    # for initializer in session.get_initializers():
    #     total_params += np.prod(initializer.shape)
    # Iterate over each initializer (parameter)
    for initializer in onnx_model.graph.initializer:
        # Add the number of elements (size) of each parameter tensor to the total
        total_params += initializer.dims[0] * initializer.dims[1] if len(initializer.dims) > 1 else initializer.dims[0]

    return total_params


def visualize_predictions(image, gt_bboxes, mask, quad, bbox, save_path=None):
    """Visualize and optionally save predictions"""
    # First, resize the mask to match the image dimensions
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    print("shape of mask while visualization",mask.shape)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    gt_image = image.copy()
    # Draw all ground truth boxes in a distinct color
    for gt_bbox in gt_bboxes:
        x1, y1, x2, y2 = gt_bbox
        cv2.rectangle(gt_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for gt
    ax[0].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Mask overlay
    overlay = image.copy()
    overlay[mask == 1] = [0, 255, 0]  # Green overlay for mask
    ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Segmentation Mask')
    ax[1].axis('off')
    
    # Quadrilateral and bbox
    result = image.copy()
    cv2.drawContours(result, [quad], 0, (0, 0, 255), 2)  # Red for quadrilateral we are getting quad from process dataset fuction
    x1, y1, x2, y2 = bbox
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for bbox
    ax[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Quadrilateral & Bounding Box')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def convert_to_coco_format(image_id, filtered_contours, scores, category_id=1):
    """Convert predictions to COCO format for evaluation"""
    results = []
    
    for i, contour in enumerate(filtered_contours):
        # Get segmentation in COCO format
        segmentation = []
        for point in contour.reshape(-1, 2):
            segmentation.extend([float(point[0]), float(point[1])])
        
        # Get bbox
        x, y, w, h = cv2.boundingRect(contour)
        
        result = {
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': [segmentation],
            'bbox': [x, y, w, h],
            'score': float(scores[i]),
            'area': float(cv2.contourArea(contour))
        }
        results.append(result)
    
    return results

def dice_coefficient(y_true, y_pred):
    """
    Calculate Dice coefficient between two binary masks
    
    Dice = 2*|X∩Y| / (|X|+|Y|)
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

def jaccard_index(y_true, y_pred):
    """
    Calculate Jaccard index (IoU) between two binary masks
    
    Jaccard = |X∩Y| / |X∪Y| = |X∩Y| / (|X|+|Y|-|X∩Y|)
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-8)

def evaluate_coco(pred_results, gt_json_path):
    """Evaluate predictions using COCO metrics"""
    # Load ground truth
    coco_gt = COCO(gt_json_path)
    
    # Create prediction dataset
    coco_dt = coco_gt.loadRes(pred_results)
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')

    coco_eval.params.catIds = [1]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Get metrics
    ap50 = coco_eval.stats[1]  # AP at IoU=0.10
    ap75 = coco_eval.stats[2]  # AP at IoU=0.75
    mAP = coco_eval.stats[0]   # AP at IoU=0.10:0.95
    
    return {
        'AP@50': ap50,
        'AP@75': ap75,
        'mAP': mAP
    }

def xywh_to_xyxy(xywh):
    """
    Converts a list of bounding boxes in XYWH format to XYXY format.
    
    Parameters:
    xywh (list of tuples): List of bounding boxes in XYWH format [(x, y, w, h), ...]
    
    Returns:
    list of tuples: List of bounding boxes in XYXY format [(x1, y1, x2, y2), ...]
    """
    xyxy = []
    
    for box in xywh:
        x, y, w, h = box
        # Calculate XYXY
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        xyxy.append((int(x1), int(y1), int(x2), int(y2)))
    
    return xyxy[-1]
def create_gt_mask_from_annotations(annots, height, width):
    """Create ground truth mask from COCO annotations"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for annot in annots:
        # Get segmentation points
        if 'segmentation' in annot and len(annot['segmentation']) > 0:
            # Handle RLE or polygon format
            if isinstance(annot['segmentation'], list):
                # Polygon format
                for seg in annot['segmentation']:
                    # Convert flat list to points array
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
            else:
                # RLE format (would need pycocotools.mask for this)
                # For simplicity, we'll use the bbox in this case
                x, y, w, h = annot['bbox']
                cv2.rectangle(mask, (int(x), int(y)), (int(x+w), int(y+h)), 1, -1)
        else:
            # Fallback to bbox if segmentation is not available
            x, y, w, h = annot['bbox']
            cv2.rectangle(mask, (int(x), int(y)), (int(x+w), int(y+h)), 1, -1)
    
    return mask

def process_dataset(onnx_model_path, image_paths, gt_json_path, output_dir, batch_size=4, threshold=0.49, use_gpu=True, save_pred: int = 50, apply_sigmoid:bool = True):
    """Process all images in dataset and evaluate with batch processing"""
    results = []
    total_time = 0
    inference_count = 0
    
    # Load COCO ground truth
    coco_gt = COCO(gt_json_path)
    
    with open(gt_json_path, "r") as fp:
        gt = json.load(fp)
        fp.close()

    image_gt = gt["images"]
    annots = gt["annotations"]
    
    # Create proper mappings
    image_gt2_id = {img_gt["file_name"] : img_gt["id"] for img_gt in image_gt}
    
    # Create a mapping from image_id to its annotations
    image_to_annots = {}
    for annot in annots:
        img_id = annot['image_id']
        if img_id not in image_to_annots:
            image_to_annots[img_id] = []
        image_to_annots[img_id].append(annot)
    
    # Create ONNX session once with GPU support
    session = get_onnx_session(onnx_model_path, use_gpu=use_gpu)
    onnx_model = onnx.load(onnx_model_path)
    # Calculate parameters
    num_params = calculate_model_params(onnx_model)
    
    # Process images in batches
    total_images = len(image_paths)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division
    
    # Set up progress bar with more details
    pbar = tqdm(total=total_images, desc="Processing images")
    
    for batch_idx in range(num_batches):
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_size = end_idx - start_idx
        
        # Load images and extract IDs
        batch_images = []
        batch_ids = []
        batch_shapes = []
        batch_file_names = []
        
        for img_path in batch_paths:
            # Extract filename
            file_name = img_path.split('/')[-1]
            batch_file_names.append(file_name)
            
            # Check if filename exists in mapping
            if file_name not in image_gt2_id:
                # print(f"Warning: File {file_name} not found in ground truth")
                continue
                
            # Get image ID
            image_id = image_gt2_id[file_name]
            batch_ids.append(image_id)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                # print(f"Warning: Could not load image {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batch_images.append(image)
            batch_shapes.append(image.shape)

        if not batch_images:
            continue
            
        # Preprocess batch
        processed_batch = batch_preprocess_images(batch_images)
        
        # Run inference on batch
        mask_logits_batch, inference_time = run_batch_inference(session, processed_batch)
        total_time += inference_time
        inference_count += len(batch_images)
        
        # Calculate and display current FPS
        current_fps = current_batch_size / inference_time
        pbar.set_postfix({"Batch": f"{batch_idx+1}/{num_batches}", "FPS": f"{current_fps:.2f}"})
        pbar.update(current_batch_size)
        
        # Postprocess batch
        binary_masks_batch = postprocess_masks(mask_logits_batch, threshold, apply_sigmoid=apply_sigmoid)
        # original_masks_batch = map_to_original_sizes(binary_masks_batch, batch_shapes, scale_infos)
        dice_scores = []
        jaccard_scores = []
        # Process each result in the batch
        for i, (image_id, image, binary_mask,mask_logit, file_name) in enumerate(zip(
                batch_ids, batch_images,binary_masks_batch, mask_logits_batch, batch_file_names)):
            
            # Extract contours and shapes
            
            # print(f"Contours found: {len(contours)} for image {file_name}")
           
            contours = extract_contours(binary_mask)
            pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for contour in contours:
                cv2.drawContours(pred_mask, [contour], 0, 1, -1)
            
            # Create ground truth mask
            if image_id in image_to_annots:
                gt_annots = image_to_annots[image_id]
                gt_mask = create_gt_mask_from_annotations(gt_annots, image.shape[0], image.shape[1])
                
                # Calculate Dice and Jaccard metrics
                dice = dice_coefficient(gt_mask, pred_mask)
                jaccard = jaccard_index(gt_mask, pred_mask)
                
                dice_scores.append(dice)
                jaccard_scores.append(jaccard)
                
                # Debug output for individual image metrics
                print(f"Image {file_name}: Dice = {dice:.4f}, Jaccard = {jaccard:.4f}")
            
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(batch_images[i])
                plt.title(f"Original Image {i}")
                
                
            # Calculate scores for contours
            # scores = []
            # for contour in contours:
            #     mask = np.zeros(batch_shapes[i][:2], dtype=np.uint8)
            #     cv2.drawContours(mask, [contour], 0, 1, -1)
            #     if apply_sigmoid:
            #         sigmoid_mask = 1 / (1 + np.exp(-mask_logit[0]))
            #     else:
            #         sigmoid_mask = mask_logit
            #    # mapped_sigmoid = map_to_original_sizes([sigmoid_mask], [batch_shapes[i]], [scale_infos[i]])[0]
            #     mean_score = np.mean(sigmoid_mask[mask > 0]) if np.sum(mask) > 0 else 0
            #     scores.append(mean_score)
            # Calculate scores for contours
            scores = []
            for contour in contours:
                # Create mask in original dimensions
                orig_mask = np.zeros(batch_shapes[i][:2], dtype=np.uint8)
                cv2.drawContours(orig_mask, [contour], 0, 1, -1)
                
                # Resize to model dimensions (1024x1024)
                model_mask = cv2.resize(orig_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                
                # Apply sigmoid if needed (using channel 1 as we determined earlier)
                if apply_sigmoid:
                    sigmoid_mask = 1 / (1 + np.exp(-mask_logit[1]))
                else:
                    sigmoid_mask = mask_logit[1]
                
                # Use properly sized mask for indexing
                mean_score = np.mean(sigmoid_mask[model_mask > 0]) if np.sum(model_mask) > 0 else 0
                scores.append(mean_score)
            
            
            # Get COCO format results
            coco_results = convert_to_coco_format(image_id,contours, scores)
            results.extend(coco_results)
            
            # Visualize a few samples
            if len(results) < save_pred:  # Save first N samples
                # Get ground truth bounding boxes for this image
                gt_bboxes = []
                if image_id in image_to_annots:
                    for annot in image_to_annots[image_id]:
                        # Convert from COCO format [x,y,w,h] to [x1,y1,x2,y2]
                        x, y, w, h = annot['bbox']
                        gt_bboxes.append((int(x), int(y), int(x+w), int(y+h)))
                
                # Debug output
                # print(f"Image: {file_name}, ID: {image_id}, GT boxes: {len(gt_bboxes)}")
                
                for j, contour in enumerate(contours):
                    quad = get_quadrilateral(contour)
                    bbox = get_bbox(contour)
                    save_path = f"{output_dir}/sample_{image_id}_{j}.png"
                    
                    # Using the image with GT already drawn
                    visualize_predictions(image, gt_bboxes, binary_mask, quad, bbox, save_path)
    
    pbar.close()
    
    # Calculate overall FPS
    fps = inference_count / total_time if total_time > 0 else 0
    
    # Evaluate using COCO metrics
    metrics = evaluate_coco(results, gt_json_path)
    metrics['FPS'] = fps
    metrics['Parameters'] = num_params
    mean_dice = np.mean(dice_scores) if dice_scores else 0
    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    metrics['Dice'] = mean_dice
    metrics['Jaccard'] = mean_jaccard
    
    return metrics, results

# Main execution
def main():
    onnx_model_path = "/home/AD/smajumder/gridaero/runway_segmentation_model13.onnx"
    input_image_dir = "/home/AD/smajumder/lard_nominal/LARDS_test/synthetic_test/images"
    test_json_path = "/home/AD/smajumder/lard_nominal/LARDS_test/synthetic_test/annotations.json"
    output_dir = "/home/AD/smajumder/lards_tests"
    batch_size = 4
    apply_sigmoid = True
    use_gpu = True  
    save_pred = 10
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # # print available providers for debugging
    # print(f"Available ONNX Runtime providers: {onnxruntime.get_available_providers()}")
    
    # Get image paths from COCO annotation file
    coco_gt = COCO(test_json_path)
    image_ids = coco_gt.getImgIds()
    images = coco_gt.loadImgs(image_ids)
    image_paths = [f"{input_image_dir}/{img['file_name']}" for img in images]
    # image_paths = glob(f"/home/AD/sthapa/grid_aero/runway_dataset/merged/data/*")
    
    # print(f"Processing {len(image_paths)} images with batch size {batch_size}")
    
    # Process dataset
    metrics, results = process_dataset(
        onnx_model_path, 
        image_paths, 
        test_json_path, 
        output_dir, 
        batch_size=batch_size,
        use_gpu=use_gpu,
        save_pred=save_pred,
        apply_sigmoid=apply_sigmoid
    )
    print(f"\nModel Performance Metrics:")
    print(f"AP@50: {metrics['AP@50']:.4f}")
    print(f"AP@75: {metrics['AP@75']:.4f}")
    print(f"Dice Coefficient: {metrics['Dice']:.4f}")
    print(f"Jaccard Index (IoU): {metrics['Jaccard']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"FPS: {metrics['FPS']:.2f}")
    print(f"Parameters: {metrics['Parameters']:,}")
    
    # Save results
    # with open(f"{output_dir}/results.json", 'w') as f:
    #     json.dump(results, f)


    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()