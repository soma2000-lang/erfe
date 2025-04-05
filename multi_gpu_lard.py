import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
from torchvision import transforms

from config_lard import (INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, NUM_SEG_CLASSES, NUM_EPOCHS, NUM_LINE_CLASSES)
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

def train(model, dataloader, device, optimizer, criterion, epoch, world_size):
    model.train()
    running_loss = 0
    counter = 0
    

    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
    for idx, batch in enumerate(tqdm(dataloader, desc=f"Training on GPU:{device}", disable=device != 0)):
        counter += 1
        images = batch['image'].to(device)
        seg_true = batch['seg_mask'].to(device)
        
        optimizer.zero_grad()
        
        model_output = model(images)
        
     
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

    loss_tensor = torch.tensor(running_loss / counter if counter > 0 else float('inf'), device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    training_loss = loss_tensor.item() / world_size
    
    if device == 0:  
        print(f"Epoch {epoch+1} - Training loss: {training_loss:.4f}")
    
    return training_loss

def calculate_dice_coefficient(seg_pred, seg_true, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before softmax
        target (torch.Tensor): Ground truth segmentation mask (B, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    seg_pred = F.sigmoid(seg_pred)
    
    intersection = (seg_pred * seg_true).sum()
    union = seg_pred.sum() + seg_true.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()

def calculate_jaccard_index(seg_pred, seg_true, smooth=1e-6):
    """
    Calculate Jaccard index (IoU) for segmentation evaluation.
    
    Args:
        seg_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) before sigmoid
        target (torch.Tensor): Ground truth segmentation mask (B, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Jaccard index (IoU)
    """
    seg_pred = F.sigmoid(seg_pred)
    
    intersection = (seg_pred * seg_true).sum()
    union = seg_pred.sum() + seg_true.sum() - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    
    return jaccard.item()

def eval(model, dataloader, device, criterion, world_size):
    model.eval()
    running_loss = 0
    counter = 0
    dice_sum = 0
    jaccard_sum = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Validation on GPU:{device}", disable=device != 0)):
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
            
            loss = criterion(seg_pred, seg_true)
            running_loss += loss.item()
            
            batch_dice = calculate_dice_coefficient(seg_pred, seg_true)
            batch_iou = calculate_jaccard_index(seg_pred, seg_true)
            
            dice_sum += batch_dice
            jaccard_sum += batch_iou

    metrics_tensor = torch.tensor([running_loss, dice_sum, jaccard_sum, counter], device=device, dtype=torch.float32)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    
    counter_total = metrics_tensor[3].item()
    
    if counter_total > 0:
        validation_loss = metrics_tensor[0].item() / counter_total
        dice_avg = metrics_tensor[1].item() / counter_total
        jaccard_avg = metrics_tensor[2].item() / counter_total
    else:
        validation_loss = float('inf')
        dice_avg = 0
        jaccard_avg = 0
    
    metrics = {
        "loss": validation_loss,
        "dice_coefficient": dice_avg,
        "jaccard_index": jaccard_avg,
    }
    

    if device == 0:
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Dice Coefficient: {dice_avg:.4f}")
        print(f"IoU (Jaccard): {jaccard_avg:.4f}")
    
    return metrics

def setup(rank, world_size):
    """Initialize distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed environment"""
    dist.destroy_process_group()

def training_loop_ddp(rank, world_size, epochs, model, train_dataset, val_dataset, batch_size, criterion_class, lr):

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
  
    setup(rank, world_size)
    

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=4
    )
    

    model = model.to(device)
    

    model = DDP(model, device_ids=[rank], output_device=rank)
    

    criterion = criterion_class()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    

    train_loss_history = []
    valid_loss_history = []
    dice_history = []
    iou_history = []
    
    best_val_loss = float('inf')
    best_dice = 0.0
    best_epoch = -1
    
    checkpoint_dir = 'checkpoints'
    if rank == 0: 
        folder_check(checkpoint_dir)
    
    # Training loop
    for epoch in range(epochs):

        train_epoch_loss = train(model, train_loader, device, optimizer, criterion, epoch, world_size)
        

        val_metrics = eval(model, val_loader, device, criterion, world_size)
        valid_epoch_loss = val_metrics["loss"]
        epoch_dice = val_metrics["dice_coefficient"]
        epoch_iou = val_metrics["jaccard_index"]
        

        if rank == 0:
            train_loss_history.append(train_epoch_loss)
            valid_loss_history.append(valid_epoch_loss)
            dice_history.append(epoch_dice)
            iou_history.append(epoch_iou)
            
            print(f"------ End of Epoch {epoch + 1}/{epochs} -------")
            

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
            
                torch.save(model.module.state_dict(), 
                           f'{checkpoint_dir}/runway_seg_epoch_{epoch}_loss_{valid_epoch_loss:.3f}_dice_{epoch_dice:.3f}_iou_{epoch_iou:.3f}.pth')
                
                torch.save(model.module.state_dict(), 'runway_segmentation_best_model.pth')
                print(f"\nModel saved at epoch: {epoch + 1} (improved {save_reason})\n")
        

        dist.barrier()
    

    if rank == 0:
        print("\nTraining complete!")

        loss_plot(train_loss_history, valid_loss_history, dice_history, iou_history)
        
        print(f"Best model at epoch {best_epoch + 1}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best Dice coefficient: {best_dice:.4f}")
    

    cleanup()

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

def main():

    world_size = 2
    

    image_dir = "/home/AD/smajumder/lard/data/"
    coordinates_csv_path = "/home/AD/smajumder/gridaero/LARD_train.csv"

    image_paths = []
    for fname in os.listdir(image_dir):
        if fname.endswith('.jpeg'):
            image_paths.append(os.path.join(image_dir, fname))
    
    line_paths = [""] * len(image_paths)
    
    train_idx = int(len(image_paths) * 0.8)  

    train_dataset = RunwayDataset(
        image_paths=image_paths[:train_idx],
        coordinates_csv_path=coordinates_csv_path,
        augment=True
    )

    val_dataset = RunwayDataset(
        image_paths=image_paths[train_idx:],
        coordinates_csv_path=coordinates_csv_path,
        augment=False
    )
    
   
    model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
    

    mp.spawn(
        training_loop_ddp,
        args=(world_size, NUM_EPOCHS, model, train_dataset, val_dataset, BATCH_SIZE, CombinedLoss, LEARNING_RATE),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system')
    main()