import torch
import os
from pathlib import Path
from ..utils.config import config

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename=None):
    if filename is None:
        filename = Path(config.CHECKPOINT_DIR) / config.CHECKPOINT_FILE

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer=None, filename=None):
    if filename is None:
        filename = Path(config.CHECKPOINT_DIR) / config.CHECKPOINT_FILE

    if not os.path.exists(filename):
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
