
import torch
INPUT_SHAPE = (3, 640, 360)  
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
NUM_SEG_CLASSES = 1
NUM_LINE_CLASSES = 5  # LEDG, REDG, CTL, AimP, THR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")