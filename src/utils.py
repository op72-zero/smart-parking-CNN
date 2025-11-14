import torch
import os
import json
import numpy as np
from datetime import datetime

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    """Load model checkpoint"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    checkpoint = torch.load(filename, map_location='cpu')
    print(f"Checkpoint loaded: {filename}")
    return checkpoint

def calculate_accuracy(outputs, labels):
    """Calculate accuracy"""
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels.data).double() / labels.size(0)

def save_training_config(config, filename):
    """Save training configuration"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_training_config(filename):
    """Load training configuration"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_timestamp():
    """Get current timestamp for logging"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0