import torch

import onnx
from model_lard import ERFE 
from config_lard import NUM_SEG_CLASSES, NUM_LINE_CLASSES, DEVICE


model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)


# model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=DEVICE))
# In onnx_convert.py, line 11, change:
checkpoint = torch.load('checkpoints/runway_seg_epoch_6_loss_0.693_dice_0.909_iou_0.833.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])  # Extract just the model part
model.eval()


dummy_input = torch.randn(1, 3, 360, 640)  


torch.onnx.export(
    model,
    dummy_input,
    "runway_segmentation_model3.onnx",
    input_names=["input"],
    output_names=["segmentation"],
    dynamic_axes={'input': {0: 'batch_size'},
                 'segmentation': {0: 'batch_size'}
                 },
    opset_version=11,
    do_constant_folding=True,
    verbose=False
)
print("Model converted successfully to ONNX format.")