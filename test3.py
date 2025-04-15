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


def get_onnx_session(onnx_model_path, use_gpu=True):
    """Create an optimized ONNX runtime session with GPU support if available"""
    
    providers = []
    print(onnxruntime.get_available_providers())
    if use_gpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print("Using CUDA for inference")
    else:
        if use_gpu:
            print("CUDA requested but not available. Falling back to CPU.")
        providers.append('CPUExecutionProvider')
        print("Using CPU for inference")
    
  
    session_options = onnxruntime.SessionOptions()
   
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        sess_options=session_options,
        providers=providers
    )
    
    return session


def preprocess_image(image, max_size=1024):

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
    

    scale = max(image.shape[0], image.shape[1]) / max_size
    pad_h = (max_size - image.shape[0] / scale) / 2
    pad_w = (max_size - image.shape[1] / scale) / 2
    
    return transformed["image"], (scale, pad_h, pad_w)

def batch_preprocess_images(images, max_size=1024):
    """Preprocess a batch of images"""
    processed_images = []
    scale_infos = []
    
    for image in images:
        processed_img, scale_info = preprocess_image(image, max_size)
        processed_images.append(processed_img)
        scale_infos.append(scale_info)
    
   
    batch_images = np.stack(processed_images, axis=0)
    
    return batch_images, scale_infos

def run_batch_inference(session, batch_images):
    """Run inference on a batch of images"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_data = batch_images.transpose(0, 3, 1, 2).astype(np.float32)
 
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = time.time() - start_time
    
    return outputs[0], inference_time

def postprocess_masks(mask_logits, threshold=0.5, apply_sigmoid: bool = False):
    """Apply sigmoid and threshold to get binary masks for a batch"""
    if apply_sigmoid:
        sigmoid_masks = 1 / (1 + np.exp(-mask_logits))
    else:
        sigmoid_masks = mask_logits
    binary_masks = (sigmoid_masks[:, 1] > threshold).astype(np.uint8)
    return binary_masks

def map_to_original_sizes(masks, original_shapes, scale_infos):
    """Map masks back to original image sizes"""
    original_masks = []
    
    for i, mask in enumerate(masks):
        scale, pad_h, pad_w = scale_infos[i]
        h, w = original_shapes[i][:2]
        roi_h_start = int(pad_h)
        roi_h_end = int(1024 - pad_h)
        roi_w_start = int(pad_w)
        roi_w_end = int(1024 - pad_w)

        cropped_mask = mask[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
        
        # Resize to original dimensions
        original_sized_mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        original_masks.append(original_sized_mask)
    
    return original_masks

def extract_contours(mask, min_area=0):  
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return contours

def get_quadrilateral(contour):
    """Extract a quadrilateral from the contour"""
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def get_bbox(contour):
    
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]  # [x1, y1, x2, y2] format

def calculate_model_params(onnx_model):
    """Calculate and return number of parameters in the model"""
    total_params = 0

    for initializer in onnx_model.graph.initializer:
   
        total_params += initializer.dims[0] * initializer.dims[1] if len(initializer.dims) > 1 else initializer.dims[0]

    return total_params

def visualize_predictions(image, gt_bboxes, mask, quad, bbox, save_path=None):

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    gt_image = image.copy()
    
    for gt_bbox in gt_bboxes:
        x1, y1, x2, y2 = gt_bbox
        cv2.rectangle(gt_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for gt
    ax[0].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image with GT Boxes')
    ax[0].axis('off')
    overlay = image.copy()
    overlay[mask == 1] = [0, 255, 0]  # Green overlay for mask
    ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Segmentation Mask')
    ax[1].axis('off')
    

    result = image.copy()
    cv2.drawContours(result, [quad], 0, (0, 0, 255), 2)  # Red for quadrilateral
    x1, y1, x2, y2 = bbox
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for bbox
    ax[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Quadrilateral & Bounding Box')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
  
    plt.close()

def convert_to_coco_format(image_id, contours, scores, category_id=1):
    """Convert predictions to COCO format for evaluation"""
    results = []
    
    for i, contour in enumerate(contours):

        segmentation = []
        contour_array = contour.reshape(-1, 2)
        for point in contour_array:
            segmentation.extend([float(point[0]), float(point[1])])
        
        
        x, y, w, h = cv2.boundingRect(contour)
        
     
        if w > 0 and h > 0 and len(segmentation) >= 6:  
            score = scores[i] if i < len(scores) else 0.5 
            
            result = {
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [segmentation],
                'bbox': [x, y, w, h],
                'score': float(score),
                'area': float(cv2.contourArea(contour))
            }
            results.append(result)
    
    return results

def evaluate_coco(pred_results, gt_json_path):
    """Evaluate predictions using COCO metrics"""
    if not pred_results:
        return {
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0
        }
    

    temp_pred_file = 'temp_predictions.json'
    with open(temp_pred_file, 'w') as f:
        json.dump(pred_results, f)
    try:
        coco_gt = COCO(gt_json_path)
        coco_dt = coco_gt.loadRes(temp_pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Get metrics
        ap50 = coco_eval.stats[1]  # AP at IoU=0.50
        ap75 = coco_eval.stats[2]  # AP at IoU=0.75
        mAP = coco_eval.stats[0]   # AP at IoU=0.50:0.95
        
        return {
            'AP@50': ap50,
            'AP@75': ap75,
            'mAP': mAP
        }
    except Exception as e:
 
        return {
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0
        }
    finally:

        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)

def process_dataset(onnx_model_path, image_paths, gt_json_path, output_dir, batch_size=8, threshold=0.5, use_gpu=True, save_pred=50, apply_sigmoid=True):
    """Process all images in dataset and evaluate with batch processing"""
    results = []
    total_time = 0
    inference_count = 0
    os.makedirs(output_dir, exist_ok=True)
   

    try:
        coco_gt = COCO(gt_json_path)
        print(f"Successfully loaded ground truth from {gt_json_path}")
    except Exception as e:

        return {
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0,
            'FPS': 0.0,
            'Parameters': 0
        }, []
    
    with open(gt_json_path, "r") as fp:
        gt = json.load(fp)
        fp.close()

    image_gt = gt["images"]
    annots = gt["annotations"]
 
    image_gt2_id = {img_gt["file_name"]: img_gt["id"] for img_gt in image_gt}

    image_to_annots = {}
    for annot in annots:
        img_id = annot['image_id']
        if img_id not in image_to_annots:
            image_to_annots[img_id] = []
        image_to_annots[img_id].append(annot)
    

    try:
        session = get_onnx_session(onnx_model_path, use_gpu=use_gpu)
        onnx_model = onnx.load(onnx_model_path)

        num_params = calculate_model_params(onnx_model)
    except Exception as e:
      
        return {
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0,
            'FPS': 0.0,
            'Parameters': 0
        }, []

    total_images = len(image_paths)
    num_batches = (total_images + batch_size - 1) // batch_size 
    
    print(f"Processing {total_images} images in {num_batches} batches")
    
 
    pbar = tqdm(total=total_images, desc="Processing images")
    
    for batch_idx in range(num_batches):
   
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_size = end_idx - start_idx
        
     
        batch_images = []
        batch_ids = []
        batch_shapes = []
        batch_file_names = []
        
        for img_path in batch_paths:
          
            file_name = os.path.basename(img_path)
            batch_file_names.append(file_name)

            if file_name not in image_gt2_id:
                continue

            image_id = image_gt2_id[file_name]
            batch_ids.append(image_id)

            image = cv2.imread(img_path)
            if image is None:
      
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batch_images.append(image)
            batch_shapes.append(image.shape)
        
     
        if not batch_images:
            continue
            
     
        processed_batch, scale_infos = batch_preprocess_images(batch_images)
        
     
        try:
            mask_logits_batch, inference_time = run_batch_inference(session, processed_batch)
            total_time += inference_time
            inference_count += len(batch_images)
    
            current_fps = current_batch_size / inference_time
            pbar.set_postfix({"Batch": f"{batch_idx+1}/{num_batches}", "FPS": f"{current_fps:.2f}"})
            pbar.update(current_batch_size)
            
            # Postprocess batch
            binary_masks_batch = postprocess_masks(mask_logits_batch, threshold, apply_sigmoid=apply_sigmoid)
            original_masks_batch = map_to_original_sizes(binary_masks_batch, batch_shapes, scale_infos)
      
            for i, (image_id, image, original_mask, mask_logit, file_name) in enumerate(zip(
                    batch_ids, batch_images, original_masks_batch, mask_logits_batch, batch_file_names)):
                
                if np.sum(original_mask) == 0:
                    print(f"Warning: Empty mask for image {file_name}")
                    continue
                
              
                contours = extract_contours(original_mask)
                print(f"Contours found: {len(contours)} for image {file_name}")
                
                if not contours:
                    
                    continue
                
                # Calculate scores for contours
                scores = []
                for contour in contours:
                    mask = np.zeros(batch_shapes[i][:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 1, -1)
                    if apply_sigmoid:
                        sigmoid_mask = 1 / (1 + np.exp(-mask_logit[0]))
                    else:
                        sigmoid_mask = mask_logit[0]  
                    
                    mapped_sigmoid = map_to_original_sizes([sigmoid_mask], [batch_shapes[i]], [scale_infos[i]])[0]
                    mean_score = np.mean(mapped_sigmoid[mask > 0]) if np.sum(mask) > 0 else 0.5
                    scores.append(mean_score)
      
                coco_results = convert_to_coco_format(image_id, contours, scores)
                if coco_results:
                    results.extend(coco_results)
        
                if save_pred > 0: 
                    gt_bboxes = []
                    if image_id in image_to_annots:
                        for annot in image_to_annots[image_id]:
                       
                            x, y, w, h = annot['bbox']
                            gt_bboxes.append((int(x), int(y), int(x+w), int(y+h)))

                    
                    for j, contour in enumerate(contours):
                        if j >= save_pred:
                            break
                            
                        quad = get_quadrilateral(contour)
                        bbox = get_bbox(contour)
                        save_path = os.path.join(output_dir, f"sample_{image_id}_{j}.png")
                        
                        
                        visualize_predictions(image, gt_bboxes, original_mask, quad, bbox, save_path)
                
        except Exception as e:
            continue
    
    pbar.close()

    fps = inference_count / total_time if total_time > 0 else 0
    

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)
  
    
    # Evaluate using COCO metrics
    if results:
        metrics = evaluate_coco(results, gt_json_path)
        metrics['FPS'] = fps
        metrics['Parameters'] = num_params
    else:
 
        metrics = {
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0,
            'FPS': fps,
            'Parameters': num_params
        }
    
    return metrics, results


def main():
    onnx_model_path = "/home/AD/smajumder/gridaero/runway_segmentation_model13.onnx"
    input_image_dir = "/home/AD/smajumder/bars/bars_test_coco/Test/data"
    test_json_path = "/home/AD/smajumder/bars/bars_test_coco/Test/annotations.json"
    output_dir = "/home/AD/smajumder/bars_tes"
 
    batch_size = 4 
    apply_sigmoid = True 
    use_gpu = True  
    save_pred = 10
    

    os.makedirs(output_dir, exist_ok=True)
    print(f"Available ONNX Runtime providers: {onnxruntime.get_available_providers()}")
    

    if not os.path.exists(input_image_dir):
        print(f"Error: Image directory {input_image_dir} does not exist")
        return
    

    coco_gt = COCO(test_json_path)
    image_ids = coco_gt.getImgIds()
   
    
    if not image_ids:
        return
    
    images = coco_gt.loadImgs(image_ids)

    
   
    image_paths = []
    for img in images:
        img_path = os.path.join(input_image_dir, img['file_name'])
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"Warning: Image file {img_path} does not exist")
    
    if not image_paths:
        print("Error: No valid image paths found")
        return
    
    
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
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"FPS: {metrics['FPS']:.2f}")
    print(f"Parameters: {metrics['Parameters']:,}")

    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f)
    

if __name__ == "__main__":
    main()