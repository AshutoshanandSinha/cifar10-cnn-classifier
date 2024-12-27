import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, target_accuracy, save_dir=None):
    """Plot training history with a professional style."""
    # Set style parameters
    plt.rcParams['figure.figsize'] = (10, 12)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 2
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='#2ecc71')
    ax1.plot(val_losses, label='Validation Loss', color='#e74c3c')
    ax1.set_title('Training and Validation Loss', pad=15)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='#2ecc71')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='#e74c3c')
    ax2.axhline(y=target_accuracy, color='#3498db', linestyle='--', 
                label=f'Target ({target_accuracy}%)')
    ax2.set_title('Training and Validation Accuracy', pad=15)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plots saved to {save_path}")
    
    plt.show() 