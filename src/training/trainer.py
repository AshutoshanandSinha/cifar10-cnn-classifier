import torch
import logging
from pathlib import Path
from ..utils.config import config
from ..utils import save_checkpoint
from .evaluator import Evaluator

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_accuracy = 0.0
        self.target_accuracy = 90.0  # You can adjust this or get from config
        self.evaluator = Evaluator(model, test_loader, criterion)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=Path(config.LOGS_DIR) / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _create_progress_bar(self, current, total, width=50):
        progress = int(width * current / total)
        return f"[{'=' * progress}{' ' * (width - progress)}]"

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (i + 1) % config.PRINT_FREQ == 0:
                msg = f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{len(self.train_loader)}], Loss: {running_loss/config.PRINT_FREQ:.4f}'
                print(msg)
                logging.info(msg)
                running_loss = 0.0

        return running_loss / len(self.train_loader)

    def train(self):
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
        print(f"Target Accuracy: {self.target_accuracy}%\n")

        for epoch in range(config.NUM_EPOCHS):
            # Training phase
            train_loss = self.train_epoch(epoch)

            # Evaluation phase
            print("\nEvaluating...")
            accuracy, eval_loss = self.evaluator.evaluate()

            # Create progress bars
            accuracy_bar = self._create_progress_bar(min(accuracy, self.target_accuracy), self.target_accuracy)
            best_accuracy_bar = self._create_progress_bar(min(self.best_accuracy, self.target_accuracy), self.target_accuracy)

            msg = f'\nEpoch {epoch+1} Summary:'
            msg += f'\n- Training Loss: {train_loss:.4f}'
            msg += f'\n- Validation Loss: {eval_loss:.4f}'
            msg += f'\n- Current Accuracy: {accuracy:.2f}% {accuracy_bar}'
            msg += f'\n- Best Accuracy:    {self.best_accuracy:.2f}% {best_accuracy_bar}'
            msg += f'\n- Target Accuracy:  {self.target_accuracy:.2f}%'

            if accuracy > self.target_accuracy:
                msg += f'\n✨ Exceeded target accuracy!'
            elif accuracy > self.best_accuracy:
                msg += f'\n🎯 New best accuracy!'

            print(msg)
            logging.info(msg)

            # Save checkpoint if it's the best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"\nSaving checkpoint...")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    loss=train_loss,
                    accuracy=accuracy
                )

        # Final summary
        print("\nTraining Complete!")
        print(f"Final Best Accuracy: {self.best_accuracy:.2f}%")
        if self.best_accuracy >= self.target_accuracy:
            print("🎉 Successfully reached target accuracy!")
        else:
            print(f"⚠️ Target accuracy not reached. Got {self.best_accuracy:.2f}% vs target {self.target_accuracy}%")
