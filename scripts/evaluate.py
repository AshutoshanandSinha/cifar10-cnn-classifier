import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

from src.models import CustomCNN
from src.data import load_data
from src.training import Evaluator
from src.utils.config import config
from src.utils import load_checkpoint

def main():
    # Setup model and load checkpoint
    model = CustomCNN()
    try:
        epoch, loss, prev_accuracy = load_checkpoint(model)
        print(f"Loaded checkpoint from epoch {epoch}")
    except FileNotFoundError as e:
        print(e)
        return

    # Setup evaluation
    _, test_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(model, test_loader, criterion)

    # Evaluate
    accuracy, avg_loss = evaluator.evaluate()

    # Print results
    print(f"\nTest Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    target_accuracy = config.TARGET_ACCURACY
    if accuracy >= target_accuracy:
        print(f"\nSuccess! Model meets the {target_accuracy}% accuracy requirement.")
    else:
        print(f"\nNote: Model has not yet achieved {target_accuracy}% accuracy.")
        print(f"Current accuracy ({accuracy:.2f}%) is {target_accuracy - accuracy:.2f}% below the target.")

if __name__ == "__main__":
    main()
