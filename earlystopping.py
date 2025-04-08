import torch
class EarlyStopping:
    """
    Early stopping to stop the training when the validation loss doesn't improve after
    a given patience.
    """
    def __init__(self, patience=7, min_delta=0, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
                            Default: 7
            min_delta (float): Minimum change in monitored quantity to qualify as improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model):
        """
        Call method to be invoked at the end of each epoch to evaluate
        if training should be stopped.
        
        Args:
            val_loss (float): Validation loss for current epoch
            model (torch.nn.Module): Model being trained
        """
        score = -val_loss  
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:  # Score did not improve enough
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # Score improved
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """
        Save model checkpoint when validation loss decreases.
        
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Model being trained
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss