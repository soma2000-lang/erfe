import torch
import torch.nn.functional as F
import torch.optim as optim
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
import argparse
import wandb
from datetime import datetime

from config_lard import (INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, NUM_SEG_CLASSES, NUM_EPOCHS, NUM_LINE_CLASSES)
from model_lard import ERFE
from loss_lard import CombinedLoss, SegmentationLoss
from dataloader_lard import RunwayDataset
from earlystopping import EarlyStopping

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

def train(model, dataloader, device, optimizer, criterion, local_rank):
    model.train()
    running_loss = 0
    counter = 0
    
    # Create progress bar only for the main process
    if local_rank == 0:
        train_iter = tqdm(enumerate(dataloader), desc="Training loop", total=len(dataloader))
    else:
        train_iter = enumerate(dataloader)
    
    for idx, batch in train_iter:
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
                if local_rank == 0:
                    print(f"ERROR: Unexpected segmentation output structure: {type(seg_output)}")
                continue
        else:
            if local_rank == 0:
                print("ERROR: Model output is not a dictionary")
            continue
            
        if not isinstance(seg_pred, torch.Tensor):
            if local_rank == 0:
                print(f"ERROR: seg_pred is not a tensor: {type(seg_pred)}")
            continue
        
        seg_pred = seg_pred.to(device)
        loss = criterion(seg_pred, seg_true)
        
        if torch.isnan(loss):
            if local_rank == 0:
                print(f"NaN detected in loss at batch {idx}")
                print("Exiting...")
            dist.destroy_process_group()
            exit(1)
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    
    # Gather loss from all processes
    training_loss = running_loss / counter if counter > 0 else float('inf')
    training_loss_tensor = torch.tensor(training_loss, device=device)
    
    # All-reduce to get average loss across all GPUs
    dist.all_reduce(training_loss_tensor, op=dist.ReduceOp.SUM)
    training_loss = training_loss_tensor.item() / dist.get_world_size()
    
    if local_rank == 0:
        print(f"Training loss: {training_loss:.4f}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
        torch.cuda.synchronize()
    
    return training_loss

def eval(model, dataloader, device, criterion, local_rank):
    model.eval()
    running_loss = 0
    counter = 0
    
    # Accumulators for global metrics
    total_intersection = 0
    total_pred_sum = 0
    total_target_sum = 0
    
    with torch.no_grad():
        # Create progress bar only for the main process
        if local_rank == 0:
            val_iter = tqdm(enumerate(dataloader), desc="Validation loop", total=len(dataloader))
        else:
            val_iter = enumerate(dataloader)
            
        for idx, batch in val_iter:
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
            
            seg_pred = seg_pred.to(device)
            loss = criterion(seg_pred, seg_true)
            running_loss += loss.item()
            
            # Apply sigmoid to predictions
            seg_pred = torch.sigmoid(seg_pred)
            
            # Calculate metrics for this batch
            intersection = (seg_pred * seg_true).sum().item()
            pred_sum = seg_pred.sum().item()
            target_sum = seg_true.sum().item()
            
            # Update accumulators
            total_intersection += intersection
            total_pred_sum += pred_sum
            total_target_sum += target_sum
    
    # Convert accumulator values to tensors for all_reduce
    metrics_tensor = torch.tensor([running_loss, total_intersection, total_pred_sum, total_target_sum], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    
    # Unpack the reduced metrics
    running_loss, total_intersection, total_pred_sum, total_target_sum = metrics_tensor.tolist()
    
    # Calculate final metrics
    validation_loss = running_loss / (counter * dist.get_world_size()) if counter > 0 else float('inf')
    smooth = 1e-6
    global_dice = (2.0 * total_intersection + smooth) / (total_pred_sum + total_target_sum + smooth)
    global_jaccard = (total_intersection + smooth) / (total_pred_sum + total_target_sum - total_intersection + smooth)
    
    metrics = {
        "loss": validation_loss,
        "dice_coefficient": global_dice,
        "jaccard_index": global_jaccard
    }
    
    if local_rank == 0:
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Global Dice: {global_dice:.4f}")
        print(f"Global Jaccard: {global_jaccard:.4f}")
    
    return metrics

def training_loop(epochs, model, train_loader, val_loader, device, optimizer, criterion, 
                  scheduler=None, local_rank=0, patience=10, min_delta=0.001, project_name="runway-segmentation",
                  run_name=None):
    # Initialize wandb only on the main process
    if local_rank == 0:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(project=project_name, name=run_name, config={
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": optimizer.__class__.__name__,
            "num_gpus": dist.get_world_size(),
            "patience": patience,
            "min_delta": min_delta
        })
    
    train_loss_history = []
    valid_loss_history = []
    dice_history = []
    iou_history = []

    checkpoint_dir = 'checkpoints'
    if local_rank == 0:
        folder_check(checkpoint_dir)
    
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            verbose=True,
            path=f'{checkpoint_dir}/best_model.pth'
        )
    
    for epoch in range(epochs):
        # Make sure all processes use the same data ordering
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
        
        if local_rank == 0:
            print(f"Epoch {epoch+1} of {epochs}")
  
        train_epoch_loss = train(model, train_loader, device, optimizer, criterion, local_rank)
        val_metrics = eval(model, val_loader, device, criterion, local_rank)
        
        valid_epoch_loss = val_metrics["loss"]
        epoch_dice = val_metrics["dice_coefficient"]
        epoch_iou = val_metrics["jaccard_index"]
       
        if local_rank == 0:
            train_loss_history.append(train_epoch_loss)
            valid_loss_history.append(valid_epoch_loss)
            dice_history.append(epoch_dice)
            iou_history.append(epoch_iou)

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_epoch_loss,
                "val_loss": valid_epoch_loss,
                "dice_coefficient": epoch_dice,
                "jaccard_index": epoch_iou,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {valid_epoch_loss:.4f}")
            print(f"Mean Dice: {epoch_dice:.4f}")
            print(f"Mean IoU: {epoch_iou:.4f}")

        if scheduler is not None:
            scheduler.step(valid_epoch_loss)
        
        # Save model only from the main process
        if local_rank == 0:
            save_model = False
            save_reason = ""
            
            best_val_loss = float('inf') if epoch == 0 else min(valid_loss_history[:-1])
            best_dice = 0.0 if epoch == 0 else max(dice_history[:-1])
            
            if valid_epoch_loss < best_val_loss:
                save_model = True
                save_reason = "validation loss"
          
            if epoch_dice > best_dice + 0.005:
                save_model = True
                save_reason = "Dice score"
                
            if save_model:
                checkpoint_path = f'{checkpoint_dir}/runway_seg_epoch_{epoch}_loss_{valid_epoch_loss:.3f}_dice_{epoch_dice:.3f}_iou_{epoch_iou:.3f}.pth'
                
                # Save both a checkpoint and the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # Save without DDP wrapper
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_epoch_loss,
                    'dice': epoch_dice,
                    'iou': epoch_iou,
                }, checkpoint_path)
                
                # Log model to wandb
                wandb.save(checkpoint_path)
                print(f"\nModel saved at epoch: {epoch + 1} (improved {save_reason})\n")
            
            # Early stopping check
            early_stopping(valid_epoch_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                # Log early stopping to wandb
                wandb.log({"early_stopped": True, "stopped_epoch": epoch})
                break
            
            print(f"Epoch {epoch + 1} completed")

    # Wait for all processes to finish
    dist.barrier()
    
    if local_rank == 0:
        # Load best model for final evaluation
        best_model_path = early_stopping.path
        
        if isinstance(model, DDP):
            # Load state dict into the module (not the DDP wrapper)
            model.module.load_state_dict(torch.load(best_model_path))
        else:
            model.load_state_dict(torch.load(best_model_path))
        
        # Final evaluation
        final_metrics = eval(model, val_loader, device, criterion, local_rank)
        print(f"Final Validation Loss: {final_metrics['loss']:.4f}")
        print(f"Final Dice Coefficient: {final_metrics['dice_coefficient']:.4f}")
        print(f"Final Jaccard Index (IoU): {final_metrics['jaccard_index']:.4f}")
        
        # Log final metrics to wandb
        wandb.log({
            "final_val_loss": final_metrics['loss'],
            "final_dice_coefficient": final_metrics['dice_coefficient'],
            "final_jaccard_index": final_metrics['jaccard_index']
        })
        
        # Plot training curves
        loss_plot(train_loss_history, valid_loss_history, dice_history, iou_history)
        
        # Finish wandb run
        wandb.finish()
    
    return model, train_loss_history, valid_loss_history, dice_history, iou_history

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
        plot_path = 'runway_segmentation_metrics.png'
        plt.savefig(plot_path)
        wandb.log({"training_curves": wandb.Image(plot_path)})
    else:
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(valid_loss, color='red', label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plot_path = 'runway_segmentation_loss.png'
        plt.savefig(plot_path)
        wandb.log({"loss_curve": wandb.Image(plot_path)})
    
    plt.close()

def setup(rank, world_size):
    """
    Initialize the distributed process group
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed process group
    """
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    """
    Main worker function
    """
    # Setup distributed process group
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Process {rank} using device: {device}")
    
    # Build datasets
    image_dir = args.image_dir
    coordinates_csv_path = args.coordinates_csv_path
    
    image_paths = []
    for fname in os.listdir(image_dir):
        image_paths.append(os.path.join(image_dir, fname))
    
    train_idx = int(len(image_paths) * 0.8)  # 80% for training
    
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
    
    # Create distributed samplers
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
    
    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
    model = model.to(device)
    
    # Wrap model with DistributedDataParallel
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Loss and optimizer
    criterion = CombinedLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    
    # Train the model
    model, train_loss, valid_loss, dice_history, iou_history = training_loop(
        epochs=args.epochs, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device, 
        optimizer=optimizer, 
        criterion=criterion,
        scheduler=scheduler,
        local_rank=rank,
        patience=args.patience,
        min_delta=args.min_delta,
        project_name=args.wandb_project,
        run_name=args.wandb_name
    )
    
    # Clean up
    cleanup()
    
    if rank == 0:
        print("Training complete!")

def main():
    parser = argparse.ArgumentParser(description='Runway Segmentation Training with Multiple GPUs and WandB')
    parser.add_argument('--image_dir', type=str, default="/home/AD/smajumder/lard/data/", 
                        help='Directory containing images')
    parser.add_argument('--coordinates_csv_path', type=str, default="/home/AD/smajumder/gridaero/LARD_train.csv", 
                        help='Path to CSV with runway coordinates')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, 
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement for early stopping')
    parser.add_argument('--wandb_project', type=str, default='runway-segmentation',
                        help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='WandB run name (default: timestamp)')
    
    args = parser.parse_args()
    
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Use multiprocessing for multi-GPU training
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Fallback to single GPU training
        print("Only one GPU detected, using single GPU training")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize wandb for single GPU
        run_name = args.wandb_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name)
        
        # Single GPU code (reusing parts of the original code)
        main_worker(0, 1, args)

if __name__ == "__main__":
 
    
    main()