# import torch

# import onnx
# from model_lard import ERFE 
# from config_lard import NUM_SEG_CLASSES, NUM_LINE_CLASSES, DEVICE


# model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)


# # model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=DEVICE))

# checkpoint = torch.load('checkpoint/runway_seg_epoch_13_loss_0.694_dice_0.011_iou_0.005.pth', map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])  
# model.eval()


# dummy_input = torch.randn(4, 3, 1024, 1024)  


# torch.onnx.export(
#     model,
#     dummy_input,
#     "runway_segmentation_model7.onnx",
#     input_names=["input"],
#     output_names=["segmentation"],
#     dynamic_axes={'input': {0: 'batch_size'},
#                  'segmentation': {0: 'batch_size'}
#                  },
#     opset_version=11,
#     do_constant_folding=True,
#     verbose=False
# )
# print("Model converted successfully to ONNX format.")
import torch
import onnx
import numpy as np
from model_lard import ERFE
from config_lard import NUM_SEG_CLASSES, NUM_LINE_CLASSES, DEVICE

# Initialize model
model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)

# Load checkpoint
checkpoint = torch.load('checkpoint/runway_seg_epoch_13_loss_0.694_dice_0.011_iou_0.005.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode and move to CPU for consistent export
model.eval()
model = model.cpu()

# Create a forward wrapper function to extract just the segmentation mask
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # Extract the segmentation output based on your model's structure
        if isinstance(output, dict) and 'segmentation' in output:
            seg_output = output['segmentation']
            if isinstance(seg_output, dict) and 'out' in seg_output:
                return seg_output['out']  # Return just the segmentation mask
            return seg_output
        return output  # If structure is different, return as is

# Wrap the model
wrapped_model = ModelWrapper(model)

# Create input with the same shape as used in training/inference
dummy_input = torch.randn(1, 3, 1024, 1024, device='cpu')

# Export the model
output_path = "runway_segmentation_model8.onnx"
torch.onnx.export(
    wrapped_model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],  # Using a simpler output name
    dynamic_axes={'input': {0: 'batch_size'},
                 'output': {0: 'batch_size'}
    },
    opset_version=11,
    do_constant_folding=True,
    verbose=False
)

# Validate the model
try:
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
except Exception as e:
    print(f"ONNX model validation failed: {e}")

print("Model converted successfully to ONNX format.")