import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall
from ..utils.config import config

class Evaluator:
    def __init__(self, model, test_loader, criterion):
        self.model = model.to(config.DEVICE)
        self.test_loader = test_loader
        self.criterion = criterion

        # Initialize metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=10).to(config.DEVICE)
        self.f1_score = F1Score(task='multiclass', num_classes=10).to(config.DEVICE)
        self.precision = Precision(task='multiclass', num_classes=10).to(config.DEVICE)
        self.recall = Recall(task='multiclass', num_classes=10).to(config.DEVICE)

    def _create_progress_bar(self, current, total, width=25):
        progress = int(width * current / total)
        return f"[{'=' * progress}{' ' * (width - progress)}]"

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        total = len(self.test_loader)
        all_predictions = []
        all_labels = []

        print("\nEvaluation Phase:")

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                # Move data to device
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Get predictions
                _, predicted = torch.max(outputs.data, 1)

                # Update metrics
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                total_loss += loss.item()

                # Store predictions and labels for metric calculation
                all_predictions.append(predicted)
                all_labels.append(labels)

                # Show progress
                progress = (i + 1) / total
                progress_bar = self._create_progress_bar(progress * 100, 100)
                print(f"\rProgress: {progress_bar} {progress*100:.1f}%", end="")

        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        accuracy = 100 * total_correct / total_samples
        avg_loss = total_loss / total
        f1 = self.f1_score(all_predictions, all_labels)
        precision = self.precision(all_predictions, all_labels)
        recall = self.recall(all_predictions, all_labels)

        # Print detailed metrics
        print("\n\nEvaluation Metrics:")
        print(f"{'=' * 40}")
        print(f"Accuracy:  {accuracy:.2f}%")
        print(f"Loss:      {avg_loss:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"{'=' * 40}")

        return accuracy, avg_loss
