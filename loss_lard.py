import torch
import torch.nn as nn
import torch.nn.functional as F
from config_lard import DEVICE
from model_lard import ERFE

class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight=2.0, device=DEVICE):
        super(SegmentationLoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight]).to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) # to handle the class imbalance problem
        self.device = device
        
    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor, but got {type(y_pred).__name__}")
        
        # For binary segmentation with LARD dataset
        # y_true shape is [batch_size, 1, height, width]
        # y_pred shape is [batch_size, 2, height, width] - we need the runway channel
        
        # Extracting runway predictions (first channel after background)
        runway_pred = y_pred[:, 1, :, :].to(self.device) #channel runway and channel for background
        runway_true = y_true[:, 0, :, :].to(self.device) 
        
       
        loss = self.bce_loss(runway_pred ,runway_true)
        return loss

class FeatureLineLoss(nn.Module):
    def __init__(self):
        super(FeatureLineLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -y_true * torch.log(y_pred)
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=2.0,device=DEVICE):
        super(CombinedLoss, self).__init__()
        self.seg_loss = SegmentationLoss(pos_weight)
        self.device = device
        
    def forward(self, y_pred, y_true):
        seg_pred = y_pred.to(self.device)
        seg_true = y_true.to(self.device)
        if not isinstance(seg_pred, torch.Tensor):
            raise TypeError(f"Expected seg_pred to be a tensor, got {type(seg_pred)}")
        seg_loss_val = self.seg_loss(seg_pred, seg_true)
        return seg_loss_val