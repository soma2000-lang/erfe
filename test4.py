import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# Import your modules
from model_lard import ERFE
from dataloader_lard import RunwayDataset
from config_lard import DEVICE, BATCH_SIZE, NUM_SEG_CLASSES, NUM_LINE_CLASSES

def folder_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def calculate_jaccard_index(seg_pred, seg_true, threshold=0.5, smooth=1e-6):
    """
    Calculate Jaccard index (IoU) with thresholding for binary segmentation.
    """
    pred_binary = (seg_pred > threshold).float()
    
    intersection = (pred_binary * seg_true).sum()
    union = pred_binary.sum() + seg_true.sum() - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    
    return jaccard.item()

def calculate_dice_coefficient(seg_pred, seg_true, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient with thresholding for binary segmentation.
    """
    pred_binary = (seg_pred > threshold).float()
    
    intersection = (pred_binary * seg_true).sum()
    union = pred_binary.sum() + seg_true.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()

def visualize_prediction(image, true_mask, pred_mask, img_name, output_dir, threshold=0.5):
    """
    Visualize the original image, ground truth mask, and predicted mask.
    """
    # Convert tensors to numpy arrays for visualization
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
    
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().squeeze().numpy()
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().squeeze().numpy()
        binary_pred_mask = (pred_mask > threshold).astype(np.float32)
    else:
        binary_pred_mask = (pred_mask > threshold).astype(np.float32)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(binary_pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_visualization.png"))
    plt.close()

def extract_contours_from_mask(mask, min_area=100):
    """
    Extract contours from a binary mask.
    """
    # Ensure mask is binary and in uint8 format
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return contours

def sort_points_clockwise(pts):
    """
    Sort points in clockwise order around their centroid.
    """
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return pts[sorted_indices]

def get_quadrilateral(contour):
    """
    Extract a quadrilateral from the largest contour.
    Uses minimum area rectangle and sorts points clockwise.
    """
    if len(contour) < 4:
        # Not enough points for a quadrilateral
        return None
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Sort points clockwise
    box = sort_points_clockwise(box)
    
    return box

def evaluate_model(model, test_loader, device, output_dir, threshold=0.5, visualize_samples=10):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    
    # Metrics
    dice_scores = []
    jaccard_scores = []
    
    # For visualization
    visualized_count = 0
    
    # Set up directories
    folder_check(output_dir)
    folder_check(os.path.join(output_dir, "visualizations"))
    folder_check(os.path.join(output_dir, "contours"))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            true_masks = batch['seg_mask'].to(device)
            img_names = batch['name']
            
            # Forward pass
            outputs = model(images)
            
            # Extract segmentation predictions
            if isinstance(outputs, dict):
                if 'segmentation' in outputs:
                    seg_output = outputs['segmentation']
                    if isinstance(seg_output, dict) and 'out' in seg_output:
                        seg_preds = seg_output['out']
                    elif isinstance(seg_output, torch.Tensor):
                        seg_preds = seg_output
                    else:
                        print(f"Unexpected segmentation output structure: {type(seg_output)}")
                        continue
                elif 'out' in outputs:
                    seg_preds = outputs['out']
                else:
                    print("Model output structure not recognized")
                    continue
            elif isinstance(outputs, torch.Tensor):
                seg_preds = outputs
            else:
                print("Model output is not a dictionary or tensor")
                continue
            
            # Calculate metrics for each image in the batch
            for i in range(images.size(0)):
                seg_pred = seg_preds[i:i+1]
                true_mask = true_masks[i:i+1]
                img_name = img_names[i]
                
                # Calculate metrics
                dice = calculate_dice_coefficient(seg_pred, true_mask, threshold)
                jaccard = calculate_jaccard_index(seg_pred, true_mask, threshold)
                
                dice_scores.append(dice)
                jaccard_scores.append(jaccard)
                
                # Visualize some samples
                if visualized_count < visualize_samples:
                    # Visualization of mask
                    visualize_prediction(
                        images[i], 
                        true_mask, 
                        seg_pred, 
                        img_name, 
                        os.path.join(output_dir, "visualizations"),
                        threshold
                    )
                    
                    # Extract and visualize contours
                    pred_mask_np = seg_pred.cpu().squeeze().numpy()
                    binary_mask = (pred_mask_np > threshold).astype(np.uint8)
                    
                    # Convert image to numpy for visualization
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    # Extract contours
                    contours = extract_contours_from_mask(binary_mask)
                    
                    if contours:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Get quadrilateral
                        quad = get_quadrilateral(largest_contour)
                        
                        if quad is not None:
                            # Draw quadrilateral on image
                            contour_img = img_rgb.copy()
                            cv2.drawContours(contour_img, [quad], 0, (0, 0, 255), 2)
                            
                            # Save image with contour
                            cv2.imwrite(
                                os.path.join(output_dir, "contours", f"{img_name}_contour.png"),
                                contour_img
                            )
                    
                    visualized_count += 1
    
    # Calculate average metrics
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
    
    # Save metrics to file
    metrics = {
        "Average Dice Coefficient": avg_dice,
        "Average Jaccard Index (IoU)": avg_jaccard
    }
    
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    print("\nEvaluation Results:")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Jaccard Index (IoU): {avg_jaccard:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Runway Segmentation Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--coordinates_csv", type=str, required=True, help="Path to coordinates CSV file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    parser.add_argument("--visualize_samples", type=int, default=10, help="Number of samples to visualize")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load model
    model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"Model loaded from {args.model_path}")
    
    # Get test images
    image_paths = []
    for fname in os.listdir(args.image_dir):
        if fname.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(args.image_dir, fname))
    
    print(f"Found {len(image_paths)} test images")
    
    # Create dataset and dataloader
    test_dataset = RunwayDataset(
        image_paths=image_paths,
        coordinates_csv_path=args.coordinates_csv,
        augment=False  # No augmentation for evaluation
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created test dataloader with {len(test_dataset)} samples")
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        threshold=args.threshold,
        visualize_samples=args.visualize_samples
    )
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()