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
from src.utils.model_analysis import analyze_model

def main():
    # Setup directories
    setup_directories()
    
    # Initialize model (let Trainer handle device placement)
    model = CustomCNN()
    
    # Analyze model architecture
    analyze_model(model)
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer)
    trainer.train()

if __name__ == "__main__":
    main()
