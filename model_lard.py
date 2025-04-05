import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import config_lard
from torch import Tensor
from typing import Dict

# so 24 is the lower channel and 112 is the higher channel
class BackboneWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        self.features = backbone
  
    def forward(self, x):
        low = None
        high = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == 4:
                low = x  # 1/8 resolution
            elif i == 12:
                high = x  # 1/16 resolution
        
        #print("Low tensor shape:", low.shape if low is not None else "None")
        #print("High tensor shape:", high.shape if high is not None else "None")
        return {"low": low, "high": high}

class Head(nn.Module):
    def __init__(self, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(
                in_channels=high_channels,
                out_channels=inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=high_channels,
                out_channels=inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.Sigmoid(),
        )
        self.high_classifier = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=num_classes+1,
            kernel_size=1
        )

    def forward(self, input_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
   
        if not isinstance(input_dict, dict):
            raise TypeError(f"Input to Head must be a dictionary with 'low' and 'high' keys, got {type(input_dict)}")
        if 'low' not in input_dict or 'high' not in input_dict:
            raise KeyError(f"Input dictionary must contain 'low' and 'high' keys, got {list(input_dict.keys())}")
        low = input_dict["low"]
        high = input_dict["high"]
        #print("Shape of low tensor in Head:", low.shape[-2:])
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        #print("Shape of x after interpolation:", x.shape)
        high_classifier = self.high_classifier(x)
        #print("Shape of output after classifier:", high_classifier.shape)
        return {"input_head_remerge_module": x, "high_classifier": high_classifier}

class ERFE(nn.Module):
    def __init__(self, num_seg_classes=config_lard.NUM_SEG_CLASSES, num_line_classes=config_lard.NUM_LINE_CLASSES):
        super().__init__()
        self.upper_conv3x3 = nn.Conv2d(in_channels=40, out_channels=num_line_classes, kernel_size=3, padding=1)
        self.upper_conv1x1 = nn.Conv2d(in_channels=num_line_classes, out_channels=num_line_classes, kernel_size=1)
        self.lower_conv3x3 = nn.Conv2d(in_channels=128, out_channels=num_line_classes, kernel_size=3, padding=1)
        self.lower_conv1x1 = nn.Conv2d(in_channels=num_line_classes, out_channels=num_line_classes, kernel_size=1)

        self.backbone = BackboneWrapper()
        self.segmentation_head = Head(high_channels=112, num_classes=num_seg_classes, inter_channels=128)
        
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.low_classifier = nn.Conv2d(40, num_seg_classes+1, 1)
        
    def get_segmentation_output(self, features):
        """
        Returns the segmentation output (seg_out) by combining high and low level predictions.
        
        Args:
            features (Dict[str, Tensor]): Dictionary containing 'low' and 'high' features
            
        Returns:
            Tensor: The segmentation output.
        """
        head_output = self.segmentation_head(features)
        high_classifier = head_output["high_classifier"]
        low_out = self.low_classifier(features["low"])
        classifier = high_classifier + low_out
        return classifier

    def forward(self, x):
    
        features = self.backbone(x)
        if not isinstance(features, dict) or 'low' not in features or 'high' not in features:
            raise ValueError("Backbone should return a dictionary with 'low' and 'high' keys") 
        low, high = features["low"], features["high"]
        #print("Shape of low tensor:", low.shape)
        #print("Shape of high tensor:", high.shape)
        head_output = self.segmentation_head(features)
        x_remerge = head_output["input_head_remerge_module"]
        #print("Shape of x_remerge:", x_remerge.shape)
        out = self.get_segmentation_output(features)
        # Resize segmentation output to match input size
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        result = OrderedDict()
        result["out"] = out
        op = self.upper_conv3x3(low)
        final_upper = self.upper_conv1x1(op)
        lower = self.lower_conv3x3(x_remerge)
        final_lower = self.lower_conv1x1(lower)
        combined = final_lower + final_upper
        p = self.adaptive_avgpool(combined)
        p = self.sigmoid(p)
        p = p.view(p.size(0), -1, 1, 1)
        return {
            'segmentation': result,
            'line_probability': p
        }
