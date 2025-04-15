
import torch
import onnx
import numpy as np
from model_lard import ERFE
from config_lard import NUM_SEG_CLASSES, NUM_LINE_CLASSES


model = ERFE(num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)


# checkpoint = torch.load('last_try/runway_seg_epoch_6_loss_0.002_dice_0.934_iou_0.877.pth', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
#checkpoint=model.load_state_dict(torch.load('last_try/best_model.pth', map_location='cpu'))
# Set to evaluation mode and move to CPU for consistent export
checkpoint = torch.load('last_try/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint) 
model.eval()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
      
        if isinstance(output, dict) and 'segmentation' in output:
            seg_output = output['segmentation']
            if isinstance(seg_output, dict) and 'out' in seg_output:
                return seg_output['out']  
            return seg_output
        return output  


wrapped_model = ModelWrapper(model)

dummy_input = torch.randn(1, 3, 1024, 1024, device='cpu')


output_path = "runway_segmentation_model14.onnx"
torch.onnx.export(
    wrapped_model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],  
    dynamic_axes={'input': {0: 'batch_size'},
                 'output': {0: 'batch_size'}
    },
    opset_version=11,
    do_constant_folding=True,
    verbose=False
)


try:
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
except Exception as e:
    print(f"ONNX model validation failed: {e}")

print("Model converted successfully to ONNX format.")