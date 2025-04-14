
import torch
INPUT_SHAPE = (3, 1024, 1024)  
BATCH_SIZE = 4
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
NUM_SEG_CLASSES = 1
NUM_LINE_CLASSES = 5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
