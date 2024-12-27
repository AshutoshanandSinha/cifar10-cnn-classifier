import torch
import logging
from pathlib import Path
from ..utils.config import config
from ..utils import save_checkpoint
from .evaluator import Evaluator
from ..utils.visualization import plot_training_history
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer):
        self.device = config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.target_accuracy = config.TARGET_ACCURACY
        self.best_accuracy = 0.0
        
        # Initialize lists to store metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Create evaluator with model on correct device
        self.evaluator = Evaluator(self.model, test_loader, criterion)
        
        # Add scheduler
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100
        )

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
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}', 
                   unit='batch', leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Ensure data and target are on the same device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        return epoch_loss, epoch_accuracy

    def train(self):
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
        print(f"Target Accuracy: {self.target_accuracy}%\n")

        for epoch in range(config.NUM_EPOCHS):
            # Training phase
            train_loss, train_accuracy = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Evaluation phase
            print("\nEvaluating...")
            accuracy, eval_loss = self.evaluator.evaluate()
            self.val_losses.append(eval_loss)
            self.val_accuracies.append(accuracy)

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
                self.best_accuracy = accuracy
                print(f"\nSaving checkpoint...")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    loss=train_loss,
                    accuracy=accuracy
                )

            # Add plotting at the end
            if epoch == config.NUM_EPOCHS - 1:
                self._plot_training_history()

            print(msg)
            logging.info(msg)

        # Final summary
        print("\nTraining Complete!")
        print(f"Final Best Accuracy: {self.best_accuracy:.2f}%")
        if self.best_accuracy >= self.target_accuracy:
            print("🎉 Successfully reached target accuracy!")
        else:
            print(f"⚠️ Target accuracy not reached. Got {self.best_accuracy:.2f}% vs target {self.target_accuracy}%")

    def _plot_training_history(self):
        """Plot and save training history."""
        from ..utils.visualization import plot_training_history
        
        plot_training_history(
            self.train_losses,
            self.val_losses,
            self.train_accuracies,
            self.val_accuracies,
            self.target_accuracy,
            save_dir=config.LOGS_DIR
        )
