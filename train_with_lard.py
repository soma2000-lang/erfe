import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config_lard import (DEVICE, INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE,NUM_SEG_CLASSES,NUM_EPOCHS,NUM_LINE_CLASSES)
from model_lard import ERFE
from loss_lard import CombinedLoss, SegmentationLoss
from dataloader_lard import RunwayDataset

def folder_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def create_mask_from_coordinates(image_shape, coordinates, class_id=1):
    """
    Create a binary mask from runway corner coordinates.
    
    Args:
        image_shape: Tuple of (height, width) for the resulting mask
        coordinates: List of [x, y] coordinates for the four corners of the runway
        class_id: Class ID to use for the runway (default: 1)
        
    Returns:
        np.ndarray: Binary mask with runway region labeled with class_id
    """
   
    mask = np.zeros(image_shape, dtype=np.uint8)
    
 
    coords_array = np.array(coordinates, dtype=np.int32)
    

    cv2.fillPoly(mask, [coords_array], class_id)
    
    return mask




def train(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0
    counter = 0
    
    for idx, batch in tqdm(enumerate(dataloader), desc="Training loop", total=len(dataloader)/BATCH_SIZE):
        counter += 1
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
            
        loss = criterion(seg_pred, seg_true)
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    
    training_loss = running_loss / counter if counter > 0 else float('inf')
    print("training loss", training_loss)
    return training_loss

def calculate_dice_coefficient(seg_pred,seg_true , smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before softmax
        target (torch.Tensor): Ground truth segmentation mask (B, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        dict: Dice coefficients for each class and mean Dice
    """

    seg_pred = F.sigmoid(seg_pred) #continuous probrability
    

    intersection = (seg_pred * seg_true).sum()
    union = seg_pred.sum() + seg_true.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    

    print("dice_coefficient",dice)
   
    
    
    return dice

def calculate_jaccard_index(seg_pred, seg_true, smooth=1e-6):
    """
    Calculate Jaccard index (IoU) for segmentation evaluation.
    
    Args:
        seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before sigmoid
        target (torch.Tensor): Ground truth segmentation mask (B, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        dict: Jaccard indices for each class and mean IoU
    """
    
    
    seg_pred = F.sigmoid(seg_pred) #continuous probrability
        

    intersection = (seg_pred * seg_true).sum()
    union = seg_pred.sum() + seg_true.sum()
    jaccard = (intersection + smooth) / (union + smooth)
        

    print("jaccard",jaccard)

def eval(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0
    counter = 0

    
    
    with torch.no_grad():
        for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
                                                        desc="Validation loop", 
                                                        total=len(dataloader)):
            counter += 1
            images = images.to(device)
            seg_true = seg_true.to(device)

            

            model_output = model(images)

            if isinstance(model_output, dict) and 'segmentation' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    continue
                    
              
          
            
            loss = criterion(seg_pred, seg_true)
            running_loss += loss.item()
            
           
            batch_dice = calculate_dice_coefficient(seg_pred, seg_true)
            batch_iou = calculate_jaccard_index(seg_pred, seg_true)
            
      
    
    validation_loss = running_loss / counter if counter > 0 else float('inf')
    
   
    
    metrics = {
        "loss": validation_loss,
        "dice_coefficient": batch_dice,
        "jaccard_index": batch_iou,
      
    }
    
    print("jaccard",batch_iou)
    print("dice_coefficient",batch_dice)
    return metrics

def training_loop(epochs, model, train_loader, val_loader, device, optimizer, criterion, scheduler=None):
    train_loss_history = []
    valid_loss_history = []
    dice_history = []
    iou_history = []

    
    best_val_loss = float('inf')
    best_dice = 0.0
    best_epoch = -1
    
    checkpoint_dir = 'checkpoints'
    folder_check(checkpoint_dir)
    
  
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
  
        # Train for one epoch
        train_epoch_loss = train(model, train_loader, device, optimizer, criterion)
        
        # Evaluate model
        val_metrics = eval(model, val_loader, device, criterion)
        valid_epoch_loss = val_metrics["loss"]
        epoch_dice = val_metrics["dice_coefficient"]
        epoch_iou = val_metrics["jaccard_index"]
       
      
        # Storing metrics history
        train_loss_history.append(train_epoch_loss)
        valid_loss_history.append(valid_epoch_loss)
        dice_history.append(epoch_dice)
        iou_history.append(epoch_iou)

        
        # Printing metrics
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")
        print(f"Mean Dice: {epoch_dice:.4f}")
        print(f"Mean IoU: {epoch_iou:.4f}")

        


        if scheduler is not None:
            scheduler.step(valid_epoch_loss)
        
     
        save_model = False
        save_reason = ""
        

        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            save_model = True
            save_reason = "validation loss"
      
        if epoch_dice > best_dice + 0.005:
            best_dice = epoch_dice
            save_model = True
            save_reason = "Dice score"
            
        if save_model:
            best_epoch = epoch
            
            torch.save(model.state_dict(), 
                       f'{checkpoint_dir}/runway_seg_epoch_{epoch}_loss_{valid_epoch_loss:.3f}_dice_{epoch_dice:.3f}_iou_{epoch_iou:.3f}.pth')

            torch.save(model.state_dict(), 'runway_segmentation_best_model.pth')
            print(f"\nModel saved at epoch: {epoch + 1} (improved {save_reason})\n")
        
        print(f"------ End of Epoch {epoch + 1} -------")

    
    
    print("\nPerforming final evaluation...")
    final_metrics = eval(model, val_loader, device, criterion)
    print(f"Validation Loss: {final_metrics['loss']:.4f}")
    print(f"Dice Coefficient: {final_metrics['dice_coefficient']:.4f}")
    print(f"Jaccard Index (IoU): {final_metrics['jaccard_index']:.4f}")
    model.load_state_dict(torch.load('runway_segmentation_best_model.pth'))
    
    return model, train_loss_history, valid_loss_history, dice_history, iou_history, best_val_loss, best_dice, best_epoch

def loss_plot(train_loss, valid_loss, dice_history=None, iou_history=None):
    """Plot and save training and validation loss curves with metrics."""
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

   
    

    image_paths = []
    for fname in os.listdir(image_dir):
        if fname.endswith('.jpeg'):
            image_paths.append(os.path.join(image_dir, fname))
    

   
    
    # Since we're dealing with semantic segmentation only, we'll use empty line paths
    # The dataloader will need to be adjusted to handle this
    #line_paths = [line_paths_dir] * len(image_paths) if os.path.exists(line_paths_dir) else [""] * len(image_paths)
    line_paths = [""] * len(image_paths)
    # Split into train and validation
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
    train_idx = int(len(image_paths) * 0.8)  # 80% for training, adjust as needed

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

    # Create data loaders
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
    print("Training complete!")





