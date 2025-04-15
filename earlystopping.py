import torch
class EarlyStopping:
 
    def __init__(self, patience=20, min_delta=0, verbose=False, path='checkpoint.pt'):
      
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_dice = None
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss,dice_score,model):
     
        score = -val_loss  
        
        if self.best_score is None or self.best_dice is None:
            self.best_score = score
            self.best_dice = dice_score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta and (dice_score < self.best_dice): 
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: 
        
          
            improvement = []
            if score >= self.best_score + self.min_delta:
                self.best_score = score
                improvement.append("validation loss")
            if dice_score >= self.best_dice:
                self.best_dice = dice_score
                improvement.append("dice score")
            
            if improvement:
                self.save_checkpoint(val_loss, model)
                if self.verbose:
                    print(f'Improvement in {" and ".join(improvement)}. Resetting counter.')
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
                