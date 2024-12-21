import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.models import CustomCNN
from src.training import Trainer
from src.utils.config import config
from src.utils import setup_directories

def main():
    # Setup directories
    setup_directories()

    # Load data
    train_loader, test_loader = load_data()

    # Initialize model, criterion, optimizer
    model = CustomCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create trainer and start training
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer)
    trainer.train()

    for epoch in range(config.NUM_EPOCHS):
        # ... epoch code ...
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint(...)

        if epoch % config.PRINT_FREQ == 0:
            print(f"Epoch [{epoch}/{config.NUM_EPOCHS}]...")

if __name__ == "__main__":
    main()
