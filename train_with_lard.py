import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

import argparse
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from earlystopping import EarlyStopping
from config_lard import (DEVICE, INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE,NUM_SEG_CLASSES,NUM_EPOCHS,NUM_LINE_CLASSES)
from model_lard import ERFE
from loss_lard import CombinedLoss, SegmentationLoss
from dataloader_lard import RunwayDataset

def folder_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def create_mask_from_coordinates(image_shape, coordinates, class_id=1):
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    
 
    coords_array = np.array(coordinates, dtype=np.int32)
    

    cv2.fillPoly(mask, [coords_array], class_id)
    
    return mask




def train(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0
    counter = 0
    
    for idx, batch in tqdm(enumerate(dataloader), desc="Training loop", total=len(dataloader)):
        counter += 1
        print(device)
        images = batch['image'].to(device)
        seg_true = batch['seg_mask'].to(device)
        
        optimizer.zero_grad()
        
        model_output = model(images)
        
        # Extract segmentation output
        if isinstance(model_output, dict) and 'segmentation' in model_output:
            seg_output = model_output['segmentation']
            if isinstance(seg_output, dict) and 'out' in seg_output:
                seg_pred = seg_output['out']
            else:
                print(f"ERROR: Unexpected segmentation output structure: {type(seg_output)}")
                continue
        else:
            print("ERROR: Model output is not a dictionary")
            continue
            
        if not isinstance(seg_pred, torch.Tensor):
            print(f"ERROR: seg_pred is not a tensor: {type(seg_pred)}")
            continue
        seg_pred = seg_pred.to(device)
        loss = criterion(seg_pred, seg_true)
        print(loss.item())
        if torch.isnan(loss):
            print(f"NaN detected in loss at batch {idx}")
            print("Exiting...")
            exit(1)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    
    training_loss = running_loss / counter if counter > 0 else float('inf')
    print("training loss", training_loss)
    print(f"After training: memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"After training: memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    torch.cuda.synchronize()  # waiting
    

  
    return training_loss

# def calculate_dice_coefficient(seg_pred,seg_true , smooth=1e-6):
#     """
#     Calculate Dice coefficient for segmentation evaluation.
    
#     Args:
#         seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before softmax
#         target (torch.Tensor): Ground truth segmentation mask (B, H, W)
#         smooth (float): Smoothing factor to avoid division by zero
        
#     Returns:
#         dict: Dice coefficients for each class and mean Dice
#     """

#     seg_pred = F.sigmoid(seg_pred) #continuous probrability
    

#     intersection = (seg_pred * seg_true).sum()
#     union = seg_pred.sum() + seg_true.sum()
#     dice = (2.0 * intersection + smooth) / (union + smooth)
    

#     print("dice_coefficient",dice)
   
    
    
#     return dice

# def calculate_jaccard_index(seg_pred, seg_true, smooth=1e-6):
#     """
#     Calculate Jaccard index (IoU) for segmentation evaluation.
    
#     Args:
#         seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before sigmoid
#         target (torch.Tensor): Ground truth segmentation mask (B, H, W)
#         smooth (float): Smoothing factor to avoid division by zero
        
#     Returns:
#         dict: Jaccard indices for each class and mean IoU
#     """
    
    
#     seg_pred = F.sigmoid(seg_pred) #continuous probrability
        

#     intersection = (seg_pred * seg_true).sum()
#     union = seg_pred.sum() + seg_true.sum()
#     jaccard = (intersection + smooth) / (union + smooth)
        

#     print("jaccard",jaccard)
def calculate_jaccard_index(seg_pred, seg_true, threshold=0.1, smooth=1e-6):
    """
    Calculate Jaccard index (IoU) with thresholding for binary segmentation.
    """


    pred_binary = (seg_pred > threshold).float()
    
    intersection = (pred_binary * seg_true).sum()
   
    union = pred_binary.sum() + seg_true.sum() - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    
    return jaccard.item()
def calculate_dice_coefficient(seg_pred, seg_true, threshold=0.1, smooth=1e-6):
    """
    Calculate Dice coefficient with thresholding for binary segmentation.
    """
   
    pred_binary = (seg_pred> threshold).float()
    
    intersection = (pred_binary * seg_true).sum()
    union = pred_binary.sum() + seg_true.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def eval(model, dataloader, device, criterion, threshold=0.1, smooth=1e-6):
    model.eval()
    running_loss = 0
    counter = 0
    
  
    total_intersection_thresholded = 0
    total_pred_sum_thresholded = 0
    total_target_sum_thresholded = 0
    total_union_thresholded = 0  
    batch_dice_values = []
    batch_jaccard_values = []
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), desc="Validation loop", total=len(dataloader)):
            counter += 1
            images = batch['image'].to(device)
            seg_true = batch['seg_mask'].to(device)
            
            model_output = model(images)
            
            if isinstance(model_output, dict) and 'segmentation' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    continue
            
 
            seg_pred = torch.sigmoid(seg_pred)
            seg_pred = seg_pred.to(device)
            
    
            loss = criterion(seg_pred, seg_true)
            running_loss += loss.item()
  
            pred_binary = (seg_pred > threshold).float()
            

            batch_intersection = (pred_binary * seg_true).sum().item()
            batch_pred_sum = pred_binary.sum().item()
            batch_target_sum = seg_true.sum().item()
            batch_union = batch_pred_sum + batch_target_sum - batch_intersection
            

            total_intersection_thresholded += batch_intersection
            total_pred_sum_thresholded += batch_pred_sum
            total_target_sum_thresholded += batch_target_sum
            total_union_thresholded += batch_union
            
      
            batch_dice = (2.0 * batch_intersection + smooth) / (batch_pred_sum + batch_target_sum + smooth)
            batch_jaccard = (batch_intersection + smooth) / (batch_union + smooth)
            
            batch_dice_values.append(batch_dice)
            batch_jaccard_values.append(batch_jaccard)

    validation_loss = running_loss / counter if counter > 0 else float('inf')
    
  
    global_dice_thresholded = (2.0 * total_intersection_thresholded + smooth) / (total_pred_sum_thresholded + total_target_sum_thresholded + smooth)
    global_jaccard_thresholded = (total_intersection_thresholded + smooth) / (total_union_thresholded + smooth)
    

    avg_batch_dice = sum(batch_dice_values) / len(batch_dice_values) if batch_dice_values else 0
    avg_batch_jaccard = sum(batch_jaccard_values) / len(batch_jaccard_values) if batch_jaccard_values else 0
    
    metrics = {
        "loss": validation_loss,
        "dice_coefficient": global_dice_thresholded,
        "jaccard_index": global_jaccard_thresholded,
        "avg_batch_dice": avg_batch_dice,
        "avg_batch_jaccard": avg_batch_jaccard
    }
    
    print(f"Average Batch Dice: {avg_batch_dice:.4f}")
    print(f"Average Batch Jaccard: {avg_batch_jaccard:.4f}")
    print(f"Global Dice (thresholded): {global_dice_thresholded:.4f}")
    print(f"Global Jaccard (thresholded): {global_jaccard_thresholded:.4f}")
    
    return metrics

def calculate_global_dice(predictions, targets, smooth=1e-6):
    """
    Calculate Dice coefficient across the entire dataset
    
    Args:
        predictions (torch.Tensor): All predicted segmentation masks
        targets (torch.Tensor): All ground truth segmentation masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Global Dice coefficient
    """
  
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def calculate_global_jaccard(predictions, targets, smooth=1e-6):
 
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection  # Correct union calculation for Jaccard
    
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard.item()
def training_loop(epochs, model, train_loader, val_loader, device, optimizer, criterion, scheduler=None, 
               patience=10, min_delta=0.001):

    train_loss_history = []
    valid_loss_history = []
    dice_history = []
    iou_history = []
    best_checkpoint = None
    
    checkpoint_dir = 'checkpoints'
    folder_check(checkpoint_dir)
    

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        path=f'{checkpoint_dir}/best_model.pth'
    )
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
  
        
        train_epoch_loss = train(model, train_loader, device, optimizer, criterion)

        val_metrics = eval(model, val_loader, device, criterion)
        valid_epoch_loss = val_metrics["loss"]
        epoch_dice = val_metrics["dice_coefficient"]
        epoch_iou = val_metrics["jaccard_index"]
      
        train_loss_history.append(train_epoch_loss)
        valid_loss_history.append(valid_epoch_loss)
        dice_history.append(epoch_dice)
        iou_history.append(epoch_iou)

        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")
        print(f"Dice Coefficient: {epoch_dice:.4f}")
        print(f"Jaccard Index (IoU): {epoch_iou:.4f}")

        if scheduler is not None:
            scheduler.step(valid_epoch_loss)
        
 
        save_model = False
        save_reason = ""
        best_val_loss = float('inf') if epoch == 0 else min(valid_loss_history[:-1])
        best_dice = 0.0 if epoch == 0 else max(dice_history[:-1])

        if valid_epoch_loss < best_val_loss:
            save_model = True
            save_reason = "validation loss"
      
        if epoch_dice > best_dice + 0.005:
            save_model = True
        #     best_checkpoint = {
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        # }
            save_reason = "Dice score"
            
        # if save_model:
        #     torch.save(model.state_dict(), 
        #               f'{checkpoint_dir}/runway_seg_epoch_{epoch}_loss_{valid_epoch_loss:.3f}._dice_{epoch_dice:.3f}_iou_{epoch_iou:.3f}.pth')
        #     print(f"\nModel saved at epoch: {epoch + 1} (improved {save_reason})\n")
        if save_model:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(checkpoint, 
                    f'{checkpoint_dir}/runway_seg_epoch_{epoch}_loss_{valid_epoch_loss:.3f}_dice_{epoch_dice:.3f}_iou_{epoch_iou:.3f}.pth')
            print(f"\nModel saved at epoch: {epoch + 1} (improved {save_reason})\n")

       
        early_stopping(valid_epoch_loss,epoch_dice,model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
        if best_checkpoint is not None:
            torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
    
 

   
    
    return model, train_loss_history, valid_loss_history, dice_history, iou_history

def loss_plot(train_loss, valid_loss, dice_history=None, iou_history=None):
 
    if dice_history is not None and iou_history is not None:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
 
        ax1.plot(train_loss, color='orange', label='train loss')
        ax1.plot(valid_loss, color='red', label='validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot metrics
        ax2.plot(dice_history, color='blue', label='Dice Coefficient')
        ax2.plot(iou_history, color='green', label='IoU (Jaccard Index)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Segmentation Quality Metrics')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('runway_segmentation_metrics.png')
    else:

        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(valid_loss, color='red', label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('runway_segmentation_loss.png')
    
    plt.close()




if  __name__ == "__main__":
   
    image_dir = "/home/AD/smajumder/lard/data/"
    coordinates_csv_path = "/home/AD/smajumder/gridaero/LARD_train.csv"

   
    parser = argparse.ArgumentParser(description='Runway Segmentation Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()

    image_paths = []
    for fname in os.listdir(image_dir):
        image_paths.append(os.path.join(image_dir, fname))
    

   
    
    # Since we're dealing with semantic segmentation only, we'll use empty line paths
    # The dataloader will need to be adjusted to handle this
    #line_paths = [line_paths_dir] * len(image_paths) if os.path.exists(line_paths_dir) else [""] * len(image_paths)
    line_paths = [""] * len(image_paths)
  
    train_idx = int(len(image_paths))
    
    # train_dataset = RunwayDataset(
    #     image_paths[:train_idx],
    #     mask_paths[:train_idx],
    #     line_paths[:train_idx],
    #     augment=True
    # )
    
    # val_dataset = RunwayDataset(
    #     image_paths[train_idx:],
    #     mask_paths[train_idx:],
    #     line_paths[train_idx:]
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    # )
    train_idx = int(len(image_paths) * 0.9)  
    # Initialize datasets
    train_dataset = RunwayDataset(
        image_paths=image_paths[:train_idx],
        coordinates_csv_path=coordinates_csv_path,
        augment=True
    )

    val_dataset = RunwayDataset(
        image_paths=image_paths[train_idx:],
        coordinates_csv_path=coordinates_csv_path,
        augment=False  # No augmentation for validation
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

    criterion = CombinedLoss(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )
    start_epoch = 0

    if args.resume is not None:
 
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

        

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
    print("Training complete!")




