import torch

import onnx
from model import ERFE 
from config import NUM_SEG_CLASSES, NUM_LINE_CLASSES, DEVICE


model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)


model.load_state_dict(torch.load('runway_segmentation_best_model1.pth', map_location=DEVICE))
model.eval()


dummy_input = torch.randn(1, 3, 360, 640)  


torch.onnx.export(
    model,
    dummy_input,
    "runway_segmentation_model1.onnx",
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