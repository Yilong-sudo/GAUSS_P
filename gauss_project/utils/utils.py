"""
Utility functions
"""

import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy.
    
    Args:
        output: Model output logits
        labels: Ground truth labels
        
    Returns:
        acc: Accuracy
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    acc = correct.sum() / len(labels)
    return acc.item()


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 200, verbose: bool = False):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Call method to check if should stop.
        
        Args:
            val_loss: Validation loss
            model: Model to save
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """
        Save model when validation loss decreases.
        
        Args:
            val_loss: Validation loss
            model: Model to save
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.val_loss_min = val_loss
    
    def load_best_model(self, model: torch.nn.Module):
        """
        Load the best model.
        
        Args:
            model: Model to load weights into
        """
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
