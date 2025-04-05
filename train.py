import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from config import (DEVICE, INPUT_SHAPE, NUM_SEG_CLASSES, NUM_LINE_CLASSES, 
                   BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS)
from model import ERFE
from loss import CombinedLoss,SegmentationLoss
from dataloader import RunwayDataset

def folder_check(folder_path):
 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# def train(model, dataloader, device, optimizer, criterion):

#     model.train()
#     running_loss = 0
#     counter = 0
    
#     for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
#                                                     desc="Training loop", 
#                                                     total=len(dataloader)):
#         counter += 1
#         images = images.to(device)
#         seg_true = seg_true.to(device)
#         line_true = line_true.to(device)
        
#         optimizer.zero_grad()
#         seg_pred, line_pred = model(images)
#         loss = criterion((seg_pred, line_pred), (seg_true, line_true))
        
#         loss.backward()
#         running_loss += loss.item()
#         optimizer.step()
    
#     training_loss = running_loss / counter
#     return training_loss
# def train(model, dataloader, device, optimizer, criterion):
#     model.train()
#     running_loss = 0
#     counter = 0
    
#     for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
#                                                     desc="Training loop", 
#                                                     total=len(dataloader)):
#         counter += 1
#         images = images.to(device)
#         seg_true = seg_true.to(device)
#         line_true = line_true.to(device)
        
#         # Add debug information
#         print(f"Images shape: {images.shape}, type: {type(images)}")
#         print(f"Seg_true shape: {seg_true.shape}, type: {type(seg_true)}")
        
#         optimizer.zero_grad()
#         seg_pred, line_pred = model(images)
        
#         # Add debug information
#         print(f"Seg_pred shape: {seg_pred.shape}, type: {type(seg_pred)}")
        
#         # Try to catch the error before it happens
#         if not isinstance(seg_pred, torch.Tensor):
#             print(f"ERROR: seg_pred is not a tensor: {type(seg_pred)}")
#             continue
            
#         loss = criterion((seg_pred, line_pred), (seg_true, line_true))
        
#         loss.backward()
#         running_loss += loss.item()
#         optimizer.step()
    
#     training_loss = running_loss / counter
#     return training_loss
# def train(model, dataloader, device, optimizer, criterion):
#     model.train()
#     running_loss = 0
#     counter = 0
    
#     for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
#                                                     desc="Training loop", 
#                                                     total=len(dataloader)):
#         counter += 1
#         images = images.to(device)
#         seg_true = seg_true.to(device)
#         line_true = line_true.to(device)
        
#         # Add debug information
#         print(f"Images shape: {images.shape}, type: {type(images)}")
#         print(f"Seg_true shape: {seg_true.shape}, type: {type(seg_true)}")
        
#         optimizer.zero_grad()
#         outputs = model(images)
        
#         # Debug the model output
#         print(f"Model output type: {type(outputs)}")
#         if isinstance(outputs, tuple):
#             print(f"Output is a tuple of length {len(outputs)}")
#             for i, out in enumerate(outputs):
#                 print(f"  Output[{i}] type: {type(out)}")
#                 if hasattr(out, 'shape'):
#                     print(f"  Output[{i}] shape: {out.shape}")
#                 else:
#                     print(f"  Output[{i}] value: {out}")
        
#         # Extract seg_pred and line_pred safely
#         if isinstance(outputs, tuple) and len(outputs) >= 2:
#             seg_pred, line_pred = outputs
#         else:
#             print("ERROR: Model output is not in expected format")
#             continue
            
#         # Check seg_pred specifically
#         if not isinstance(seg_pred, torch.Tensor):
#             print(f"ERROR: seg_pred is not a tensor: {type(seg_pred)}")
#             if isinstance(seg_pred, str):
#                 print(f"seg_pred string value: {seg_pred}")
#             continue
            
#         loss = criterion((seg_pred, line_pred), (seg_true, line_true))
        
#         loss.backward()
#         running_loss += loss.item()
#         optimizer.step()
    
#     training_loss = running_loss / counter if counter > 0 else float('inf')
#     return training_loss
# def train(model, dataloader, device, optimizer, criterion):
#     model.train()
#     running_loss = 0
#     counter = 0
    
#     for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
#                                                     desc="Training loop", 
#                                                     total=len(dataloader)):
#         counter += 1
#         images = images.to(device)
#         seg_true = seg_true.to(device)
#         line_true = line_true.to(device)
        
#         # Add debug information
#         print(f"Images shape: {images.shape}, type: {type(images)}")
#         print(f"Seg_true shape: {seg_true.shape}, type: {type(seg_true)}")
        
#         optimizer.zero_grad()
#         outputs = model(images)
        
#         # Debug the model output
#         print(f"Model output type: {type(outputs)}")
#         if isinstance(outputs, tuple):
#             print(f"Output is a tuple of length {len(outputs)}")
#             for i, out in enumerate(outputs):
#                 print(f"  Output[{i}] type: {type(out)}")
#                 if hasattr(out, 'shape'):
#                     print(f"  Output[{i}] shape: {out.shape}")
#                 else:
#                     print(f"  Output[{i}] value: {out}")
        
#         # Extract seg_pred and line_pred safely
#         if isinstance(outputs, tuple) and len(outputs) >= 2:
#             seg_pred, line_pred = outputs
#         else:
#             print("ERROR: Model output is not in expected format")
#             continue
            
#         # Check seg_pred specifically
#         if not isinstance(seg_pred, torch.Tensor):
#             print(f"ERROR: seg_pred is not a tensor: {type(seg_pred)}")
#             if isinstance(seg_pred, str):
#                 print(f"seg_pred string value: {seg_pred}")
#             continue
            
#         loss = criterion((seg_pred, line_pred), (seg_true, line_true))
        
#         loss.backward()
#         running_loss += loss.item()
#         optimizer.step()
    
#     training_loss = running_loss / counter if counter > 0 else float('inf')
#     return training_loss
def train(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0
    counter = 0
    
    for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
                                                    desc="Training loop", 
                                                    total=len(dataloader)/BATCH_SIZE):
        counter += 1
        images = images.to(device)
        seg_true = seg_true.to(device)
        line_true = line_true.to(device)
        
        optimizer.zero_grad()
        
    
        model_output = model(images)
  
        print(f"Model output type: {type(model_output)}")
        if isinstance(model_output, dict):
            print(f"Model output keys: {model_output.keys()}")
            
            # Extracting segmentation output
            if 'segmentation' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    print(f"ERROR: Unexpected segmentation output structure: {type(seg_output)}")
                    continue
            else:
                print("ERROR: No 'segmentation' key in model output")
                continue
                
            # Extract line probability output
            if 'line_probability' in model_output:
                line_pred = model_output['line_probability']
            else:
                print("ERROR: No 'line_probability' key in model output")
                continue
        else:
            print(f"ERROR: Model output is not a dictionary: {type(model_output)}")
            continue
            

        if not isinstance(seg_pred, torch.Tensor):
            print(f"ERROR: seg_pred is not a tensor: {type(seg_pred)}")
            continue
            

        loss = criterion(seg_pred, seg_true)
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    
    training_loss = running_loss / counter if counter > 0 else float('inf')
    print("training loss",training_loss)
    return training_loss
# def eval(model, dataloader, device, criterion):

#     model.eval()
#     running_loss = 0
#     counter = 0
    
#     with torch.no_grad():
#         for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
#                                                         desc="Validation loop", 
#                                                         total=len(dataloader)):
#             counter += 1
#             images = images.to(device)
#             seg_true = seg_true.to(device)
#             line_true = line_true.to(device)
            
#             seg_pred, line_pred = model(images)
#             loss = criterion((seg_pred, line_pred), (seg_true, line_true))
            
#             running_loss += loss.item()
    
#     validation_loss = running_loss / counter
#     return validation_loss
def eval(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0
    counter = 0
    
    with torch.no_grad():
        for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
                                                        desc="Validation loop", 
                                                        total=len(dataloader)/BATCH_SIZE):
            counter += 1
            images = images.to(device)
            seg_true = seg_true.to(device)
            line_true = line_true.to(device)
            
            # Get model output (now handling dictionary return type)
            model_output = model(images)
            
            # Extract outputs
            if isinstance(model_output, dict) and 'segmentation' in model_output and 'line_probability' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    continue
                    
                line_pred = model_output['line_probability']
            else:
                continue
            
            loss = criterion(seg_pred, seg_true)
            running_loss += loss.item()
    
    validation_loss = running_loss / counter if counter > 0 else float('inf')
    return validation_loss

def training_loop(epochs, model, train_loader, val_loader, device, optimizer, criterion, scheduler=None):
 
    
    train_loss_history = []
    valid_loss_history = []
    best_val_loss = float('inf')
    best_epoch = -1
    

    checkpoint_dir = 'checkpoints'
    folder_check(checkpoint_dir)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
  
        train_epoch_loss = train(model, train_loader, device, optimizer, criterion)
        valid_epoch_loss = eval(model, val_loader, device, criterion)
      
        train_loss_history.append(train_epoch_loss)
        valid_loss_history.append(valid_epoch_loss)
        
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")
        

        if scheduler is not None:
            scheduler.step(valid_epoch_loss)
        
        # Saving best model
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'{checkpoint_dir}/runway_segmentation_epoch_{epoch}_valLoss_{valid_epoch_loss:.3f}.pth')
            torch.save(model.state_dict(), 'runway_segmentation_best_model.pth')
            print(f"\nModel saved at epoch: {epoch + 1} \n")
        
        print(f"------ End of Epoch {epoch + 1} -------")
    
    # Load best model
    model.load_state_dict(torch.load('runway_segmentation_best_model.pth'))
    
    return model, train_loss_history, valid_loss_history, best_val_loss, best_epoch

def loss_plot(train_loss, valid_loss):
    """Plot and save training and validation loss curves."""
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('runway_segmentation_loss.png')
    plt.close()

def evaluate_tp_tn_rate(model, val_loader, device, threshold=0.1):
    """
    Evaluate the segmentation performance with TP+TN rate as described in the paper.
    This calculates the rate of correctly classified pixels (both true positives and true negatives).
    """
    model.eval()
    
    # Define class names for printing results
    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
    
    # Initialize counters for each class
    class_stats = {cls: {'total_pixels': 0, 'correct_pixels': 0} for cls in range(NUM_SEG_CLASSES)}
    
    # For line detection if applicable
    line_stats = {cls: {'total_samples': 0, 'correct_samples': 0} for cls in range(NUM_LINE_CLASSES)} if NUM_LINE_CLASSES > 0 else {}
    
    with torch.no_grad():
        # Process segmentation metrics
        for images, (seg_true, line_true) in tqdm(val_loader, desc="Evaluating TP+TN Rate"):
            images = images.to(device)
            seg_true = seg_true.to(device)
            
            # Get model predictions
            seg_pred, line_pred = model(images)
            
            # Process segmentation predictions
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred_classes = torch.argmax(seg_pred, dim=1)
            
            # Calculate TP+TN for each class in the batch
            # batch_size = images.size(0)
            
            # for i in range(batch_size):
            for cls in range(NUM_SEG_CLASSES):
                    # Getting binary masks for this class  we could have used the rle also inplace of this
                    true_mask = (seg_true[i] == cls)
                    pred_mask = (seg_pred_classes[i] == cls)
                    
                    # Calculating TP (correctly predicted class pixels)
                    tp = torch.logical_and(true_mask, pred_mask).sum().item()
                    
                    # Calculating  TN (correctly predicted non-class pixels)
                    tn = torch.logical_and(~true_mask, ~pred_mask).sum().item()
                    
                  
                    class_stats[cls]['total_pixels'] += true_mask.numel()
                    class_stats[cls]['correct_pixels'] += (tp + tn)
            
            # Process line predictions if applicable and if line_true is provided
            # if NUM_LINE_CLASSES > 0 and line_true is not None:
            #     # Threshold the line probability maps
            #     for cls in range(NUM_LINE_CLASSES):
            #         for i in range(batch_size):
            #             # Check if line exists (max value exceeds threshold)
            #             if hasattr(line_true[i, cls], 'max'):  # Check if it's a tensor
            #                 ground_truth_line = line_true[i, cls].max() > threshold
            #                 line_detected = line_pred[i, cls].max() > threshold
                            
            #                 line_stats[cls]['total_samples'] += 1
                            
            #                 # Count TP and TN
            #                 if (line_detected and ground_truth_line) or (not line_detected and not ground_truth_line):
            #                     line_stats[cls]['correct_samples'] += 1
    
    # Calculate TP+TN rates for segmentation
    class_tp_tn_rates = {}
    for cls in range(NUM_SEG_CLASSES):
        if class_stats[cls]['total_pixels'] > 0:
            tp_tn_rate = class_stats[cls]['correct_pixels'] / class_stats[cls]['total_pixels']
            class_tp_tn_rates[cls] = tp_tn_rate
    
    # Calculate TP+TN rates for line detection not required
    # line_tp_tn_rates = {}
    # if NUM_LINE_CLASSES > 0:
    #     line_class_names = ["Left Edge", "Center Line", "Right Edge", "Aiming Point", "Threshold"]
    #     for cls in range(NUM_LINE_CLASSES):
    #         if cls in line_stats and line_stats[cls]['total_samples'] > 0:
    #             tp_tn_rate = line_stats[cls]['correct_samples'] / line_stats[cls]['total_samples']
    #             line_tp_tn_rates[cls] = tp_tn_rate
    
    # Print results
    print("\nSegmentation TP+TN Rates:")
    for cls in range(NUM_SEG_CLASSES):
        if cls in class_tp_tn_rates:
            print(f"{class_names[cls]}: {class_tp_tn_rates[cls]:.4f}")
    
    # if NUM_LINE_CLASSES > 0 and line_tp_tn_rates:
    #     print("\nLine Detection TP+TN Rates:")
    #     for cls in range(NUM_LINE_CLASSES):
    #         if cls in line_tp_tn_rates:
    #             line_name = line_class_names[cls] if cls < len(line_class_names) else f"Line {cls}"
    #             print(f"{line_name}: {line_tp_tn_rates[cls]:.4f}")
    
    # Calculate overall rates
    overall_seg_tp_tn_rate = sum(class_tp_tn_rates.values()) / len(class_tp_tn_rates) if class_tp_tn_rates else 0
    # overall_line_tp_tn_rate = sum(line_tp_tn_rates.values()) / len(line_tp_tn_rates) if line_tp_tn_rates else 0
    
    print(f"\nOverall Segmentation TP+TN Rate: {overall_seg_tp_tn_rate:.4f}")
    # if NUM_LINE_CLASSES > 0 and line_tp_tn_rates:
    #     print(f"Overall Line Detection TP+TN Rate: {overall_line_tp_tn_rate:.4f}")
    
    return class_tp_tn_rates, line_tp_tn_rates

def visualize_tp_tn_rates(class_tp_tn_rates, line_tp_tn_rates=None, epoch=None):
    """
    Visualize TP+TN rates using bar charts similar to the paper's figures.
    """
    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
    line_class_names = ["Left Edge", "Center Line", "Right Edge", "Aiming Point", "Threshold"]
    
    # Set up the figure
    num_plots = 2 if line_tp_tn_rates else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]  # Make it iterable for single plot case
    
    # Plot segmentation TP+TN rates
    ax = axes[0]
    classes = list(class_tp_tn_rates.keys())
    rates = [class_tp_tn_rates[cls] for cls in classes]
    class_labels = [class_names[cls] for cls in classes]
    
    bars = ax.bar(class_labels, rates, color='skyblue')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('TP+TN Rate')
    ax.set_title(f'Segmentation TP+TN Rates{" (Epoch "+str(epoch)+")" if epoch else ""}')
    
    # Add rate values on top of the bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # # Plot line detection TP+TN rates if available
    # if line_tp_tn_rates:
    #     ax = axes[1]
    #     line_classes = list(line_tp_tn_rates.keys())
    #     line_ates = [line_tp_tn_rates[cls] for cls in line_classes]
    #     line_labels = [line_class_names[cls] if cls < len(line_class_names) else f"Line {cls}" 
    #                   for cls in line_classes]
        
    #     bars = ax.bar(line_labels, line_rates, color='lightgreen')
    #     ax.set_ylim(0, 1.0)
    #     ax.set_ylabel('TP+TN Rate')
    #     ax.set_title(f'Line Detection TP+TN Rates{" (Epoch "+str(epoch)+")" if epoch else ""}')
        
    #     # Add rate values on top of the bars
    #     for bar, rate in zip(bars, line_rates):
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
    #                 f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_name = f'tp_tn_rates{"_epoch_"+str(epoch) if epoch else ""}.png'
    plt.savefig(save_name)
    plt.close()
    
    return save_name
def calculate_dice_coefficient(seg_pred, seg_true, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before softmax
        seg_true (torch.Tensor): Ground truth segmentation mask (B, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        dict: Dice coefficients for each class and mean Dice
    """
    num_classes = 4
    dice_scores = {}
    

    seg_pred = F.softmax(seg_pred, dim=1)
    

    pred_classes = torch.argmax(seg_pred, dim=1)
    
    for cls in range(num_classes):
        pred_cls = (pred_classes == cls).float()
        target_cls = (seg_true == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
       
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[f"class_{cls}"] = dice.item()
    

    class_dice_values = [dice_scores[f"class_{cls}"] for cls in range(num_classes)]
    dice_scores["mean"] = sum(class_dice_values) / len(class_dice_values)
    
    return dice_scores

def calculate_jaccard_index(seg_pred, seg_true, smooth=1e-6):
    """
    Calculate Jaccard index (IoU) for segmentation evaluation.
    
    Args:
        seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before softmax
        seg_true (torch.Tensor): Ground truth segmentation mask (B, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        dict: Jaccard indices for each class and mean IoU
    """
    num_classes = 4
    iou_scores = {}
    seg_pred = F.softmax(seg_pred, dim=1)
    pred_classes = torch.argmax(seg_pred, dim=1)
    

    for cls in range(num_classes):
        pred_cls = (pred_classes == cls).float()
        target_cls = (seg_true == cls).float() #can cls be same
        
        intersection = (pred_cls * target_cls).sum()
        # For IoU, the union is (A+B-intersection)
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        # IoU : intersection / union
        iou = (intersection + smooth) / (union + smooth)
        iou_scores[f"class_{cls}"] = iou.item()
    
    # Calculating  mean IoU (class-wise average)
    class_iou_values = [iou_scores[f"class_{cls}"] for cls in range(num_classes)]
    iou_scores["mean"] = sum(class_iou_values) / len(class_iou_values)
    
    return iou_scores

def evaluate_segmentation_metrics(model, dataloader, device, class_names):
    """
    Evaluate segmentation model using multiple metrics:
    - Dice coefficient
    - Jaccard index (IoU)
    - TP+TN rate
    
    Args:
        model (nn.Module): Segmentation model
        dataloader (DataLoader): Validation dataloader
        device (torch.device): Device to run evaluation on
        class_names (list): List of class names for reporting
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()

    class_dice_values = {cls: [] for cls in range(len(class_names))}
    class_iou_values = {cls: [] for cls in range(len(class_names))}
    

    class_tp_tn = {cls: {'total_pixels': 0, 'correct_pixels': 0} for cls in range(len(class_names))}
    
    with torch.no_grad():
        for images, batch_targets in tqdm(dataloader, desc="Evaluating metrics"):
  
            seg_true = batch_targets[0]
           
            
            images = images.to(device)
            seg_true = seg_true.to(device)
            
            model_output = model(images)
            
            # Extract segmentation output
            if isinstance(model_output, dict) and 'segmentation' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    print("Error: Unexpected segmentation output structure")
                    continue
            else:
                print("Error: Model output structure not as expected")
                continue

            batch_dice = calculate_dice_coefficient(seg_pred, seg_true)
            batch_iou = calculate_jaccard_index(seg_pred, seg_true)
        
            for cls in range(len(class_names)):
                class_dice_values[cls].append(batch_dice[f"class_{cls}"])
                class_iou_values[cls].append(batch_iou[f"class_{cls}"])
            
          
            # seg_pred_softmax = F.softmax(seg_pred, dim=1)
            # seg_pred_classes = torch.argmax(seg_pred_softmax, dim=1)
           
    

    dice_scores = {}
    iou_scores = {}
    

    for cls in range(len(class_names)):
        dice_scores[f"class_{cls}"] = sum(class_dice_values[cls]) / len(class_dice_values[cls]) if class_dice_values[cls] else 0
        iou_scores[f"class_{cls}"] = sum(class_iou_values[cls]) / len(class_iou_values[cls]) if class_iou_values[cls] else 0
    

    dice_scores["mean"] = sum(dice_scores.values()) / len(class_names)
    iou_scores["mean"] = sum(iou_scores.values()) / len(class_names)
    
    

    # tp_tn_rates = {}
    # for cls in range(len(class_names)):
    #     if class_tp_tn[cls]['total_pixels'] > 0:
    #         tp_tn_rate = class_tp_tn[cls]['correct_pixels'] / class_tp_tn[cls]['total_pixels']
    #         tp_tn_rates[f"class_{cls}"] = tp_tn_rate
    
    # tp_tn_rates["mean"] = sum(tp_tn_rates.values()) / len(tp_tn_rates)
    
    # Combine all metrics
    metrics = {
        "dice_coefficient": dice_scores,
        "jaccard_index": iou_scores,
        # "tp_tn_rate": tp_tn_rates
    }
    
 
 
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name}: {dice_scores[f'class_{i}']:.4f}")
    print(f"Mean Dice: {dice_scores['mean']:.4f}")
    

    for i, cls_name in enumerate(class_names):
        print(f"{cls_name}: {iou_scores[f'class_{i}']:.4f}")
    print(f"Mean IoU: {iou_scores['mean']:.4f}")
    

   
    
    
    
    return metrics

def visualize_segmentation_results(model, val_loader, device, num_samples=3):
    """
    Visualize segmentation predictions for a few validation samples.
    """
    model.eval()
    
    # Getting a few samples from validation set
    samples = []
    with torch.no_grad():
        for images, (seg_true, _) in val_loader:
            if len(samples) >= num_samples:
                break
            
  
            batch_size = min(num_samples - len(samples), images.size(0))
            for i in range(batch_size):
                samples.append((images[i:i+1], seg_true[i:i+1]))
    
  
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    # Class colors for visualization a given in the dataset runaway fs
    colors = [
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Runway Area - red
        [255, 255, 0],    # Aiming Point Marking - yellow
        [0, 255, 0]       # Threshold Marking - green
    ]
    colors = np.array(colors)
    
    with torch.no_grad():
        for i, (image, seg_true) in enumerate(samples):
            image = image.to(device)
            seg_true = seg_true.to(device)
            

            seg_pred, _ = model(image)
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred = torch.argmax(seg_pred, dim=1)
            image = image.cpu().numpy()[0]
            image = np.transpose(image, (1, 2, 0))
            # Denormalize image
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            
            seg_true = seg_true.cpu().numpy()[0]
            seg_pred = seg_pred.cpu().numpy()[0]
            
            # Create colored segmentation maps
            true_colored = np.zeros((seg_true.shape[0], seg_true.shape[1], 3), dtype=np.uint8)
            pred_colored = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
            
            for cls in range(NUM_SEG_CLASSES):
                true_colored[seg_true == cls] = colors[cls]
                pred_colored[seg_pred == cls] = colors[cls]
            
            # Plot
            axs[i, 0].imshow(image)
            axs[i, 0].set_title('Input Image')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(true_colored)
            axs[i, 1].set_title('Ground Truth')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(pred_colored)
            axs[i, 2].set_title('Prediction')
            axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png')
    plt.close()
def evaluate_tp_tn_rate(model, val_loader, device, threshold=0.1):
    """
    Evaluate the segmentation performance with TP+TN rate as described in the paper.
    This calculates the rate of correctly classified pixels (both true positives and true negatives).
    """
    model.eval()
    

    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
    
  
    class_stats = {cls: {'total_pixels': 0, 'correct_pixels': 0} for cls in range(NUM_SEG_CLASSES+1)}

    line_stats = {cls: {'total_samples': 0, 'correct_samples': 0} for cls in range(NUM_LINE_CLASSES)} if NUM_LINE_CLASSES > 0 else {}
    
    with torch.no_grad():

        for images, (seg_true, line_true) in tqdm(val_loader, desc="Evaluating TP+TN Rate"):
            images = images.to(device)
            seg_true = seg_true.to(device)
            
            # Get model predictions
            model_output = model(images)
            
            # Extract segmentation and line prediction from model output
            if isinstance(model_output, dict):
                if 'segmentation' in model_output and 'line_probability' in model_output:
                    seg_output = model_output['segmentation']
                    if isinstance(seg_output, dict) and 'out' in seg_output:
                        seg_pred = seg_output['out']
                    else:
                        print("Error: Unexpected segmentation output structure")
                        continue
                        
                    line_pred = model_output['line_probability']
                else:
                    print("Error: Missing keys in model output")
                    continue
            else:
                print("Error: Model output is not a dictionary")
                continue
            
         
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred_classes = torch.argmax(seg_pred, dim=1)
            
           
            # batch_size = images.size(0)
            
            # for i in range(batch_size):
            for cls in range(NUM_SEG_CLASSES):
                 
                    true_mask = (seg_true == cls)
                    pred_mask = (seg_pred_classes == cls)
                    
                   
                    tp = torch.logical_and(true_mask, pred_mask).sum().item()
                    
               
                    tn = torch.logical_and(~true_mask, ~pred_mask).sum().item()
                    
                    class_stats[cls]['total_pixels'] += true_mask.numel()
                    class_stats[cls]['correct_pixels'] += (tp + tn)
            
         
            # if NUM_LINE_CLASSES > 0 and line_true is not None:
         
            #     for cls in range(NUM_LINE_CLASSES):
            #         for i in range(batch_size):
            #             # Check if line exists (max value exceeds threshold)
            #             if hasattr(line_true[i, cls], 'max'):  # Check if it's a tensor
            #                 ground_truth_line = line_true[i, cls].max() > threshold
            #                 line_detected = line_pred[i, cls].max() > threshold
                            
            #                 line_stats[cls]['total_samples'] += 1
                            
            #                 # Count TP and TN
            #                 if (line_detected and ground_truth_line) or (not line_detected and not ground_truth_line):
            #                     line_stats[cls]['correct_samples'] += 1
    
    # Calculate TP+TN rates for segmentation
    class_tp_tn_rates = {}
    for cls in range(NUM_SEG_CLASSES):
        if class_stats[cls]['total_pixels'] > 0:
            tp_tn_rate = class_stats[cls]['correct_pixels'] / class_stats[cls]['total_pixels']
            class_tp_tn_rates[cls] = tp_tn_rate
    
    # Calculate TP+TN rates for line detection
    # line_tp_tn_rates = {}
    # if NUM_LINE_CLASSES > 0:
    #     line_class_names = ["Left Edge", "Center Line", "Right Edge", "Aiming Point", "Threshold"]
    #     for cls in range(NUM_LINE_CLASSES):
    #         if cls in line_stats and line_stats[cls]['total_samples'] > 0:
    #             tp_tn_rate = line_stats[cls]['correct_samples'] / line_stats[cls]['total_samples']
    #             line_tp_tn_rates[cls] = tp_tn_rate
    
    # Print results
    print("\nSegmentation TP+TN Rates:")
    for cls in range(NUM_SEG_CLASSES):
        if cls in class_tp_tn_rates:
            print(f"{class_names[cls]}: {class_tp_tn_rates[cls]:.4f}")
    
    # if NUM_LINE_CLASSES > 0 and line_tp_tn_rates:
    #     print("\nLine Detection TP+TN Rates:")
    #     for cls in range(NUM_LINE_CLASSES):
    #         if cls in line_tp_tn_rates:
    #             line_name = line_class_names[cls] if cls < len(line_class_names) else f"Line {cls}"
    #             print(f"{line_name}: {line_tp_tn_rates[cls]:.4f}")
    
    # Calculate overall rates
    overall_seg_tp_tn_rate = sum(class_tp_tn_rates.values()) / len(class_tp_tn_rates) if class_tp_tn_rates else 0
    # overall_line_tp_tn_rate = sum(line_tp_tn_rates.values()) / len(line_tp_tn_rates) if line_tp_tn_rates else 0
    
    print(f"\nOverall Segmentation TP+TN Rate: {overall_seg_tp_tn_rate:.4f}")
    # if NUM_LINE_CLASSES > 0 and line_tp_tn_rates:
    #     print(f"Overall Line Detection TP+TN Rate: {overall_line_tp_tn_rate:.4f}")
    
    return class_tp_tn_rates

if __name__ == "__main__":
   
    image_dir = "/home/AD/smajumder/runaway/640x360_dataset/640x360/train"
    mask_dir = "/home/AD/smajumder/runaway/resizded_trainimages_640x360"
    line_paths_dir = "/home/AD/smajumder/runaway/train_labels_640x360.json"
    

    image_paths = []
    for fname in os.listdir(image_dir):
        if fname.endswith('.png'):
            image_paths.append(os.path.join(image_dir, fname))
    

    mask_paths = []
    valid_image_paths = []
    
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)

        mask_name = img_name
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
            valid_image_paths.append(img_path)
        else:
            print(f"Warning: Mask not found for {img_path}")

    image_paths = valid_image_paths
    
    # Since we're dealing with semantic segmentation only, we'll use empty line paths
    # The dataloader will need to be adjusted to handle this
    #line_paths = [line_paths_dir] * len(image_paths) if os.path.exists(line_paths_dir) else [""] * len(image_paths)
    line_paths = [""] * len(image_paths)
    # Split into train and validation
    train_idx = int(0.9 * len(image_paths))
    
    train_dataset = RunwayDataset(
        image_paths[:train_idx],
        mask_paths[:train_idx],
        line_paths[:train_idx],
        augment=True
    )
    
    val_dataset = RunwayDataset(
        image_paths[train_idx:],
        mask_paths[train_idx:],
        line_paths[train_idx:]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    
    
    model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
    model = model.to(DEVICE)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    

    model, train_loss, valid_loss, best_val_loss, best_epoch = training_loop(
        epochs=NUM_EPOCHS, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=DEVICE, 
        optimizer=optimizer, 
        criterion=criterion,
        scheduler=scheduler
    )

    loss_plot(train_loss, valid_loss)

    class_tp_tn_rates= evaluate_tp_tn_rate(model, val_loader, DEVICE)
    evaluate_segmentation_metrics(model, val_loader, DEVICE, class_names = ["Background", "Runway Area","Aiming Point Marking","Threshold Marking"])
    visualize_tp_tn_rates(class_tp_tn_rates)
    

    # visualize_segmentation_results(model, val_loader, DEVICE, num_samples=3)
    
    print("Training complete!")