# CIFAR-10 Image Classification Project

This project implements a custom CNN architecture for classifying images from the CIFAR-10 dataset using PyTorch.

## Features

- Custom CNN architecture with:
  - Depthwise Separable Convolution
  - Dilated Convolution
  - Strided Convolutions (instead of MaxPooling)
  - Global Average Pooling
  - Total Receptive Field > 44
  - Under 200k parameters
- Data augmentation using Albumentations
- On-the-fly mean and standard deviation calculation
- Checkpoint management and model saving
- Training progress logging with visual progress bars
- YAML-based configuration
- GPU support

## Project Structure

```
cifar10_project/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/            # Model definitions
│   │   ├── __init__.py
│   │   └── cnn.py
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── checkpoints.py
│   │   └── setup.py
│   └── training/          # Training related code
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluator.py
├── config/                # Configuration files
│   └── config.yaml
├── scripts/               # Execution scripts
│   ├── train.py
│   └── evaluate.py
├── data/                  # Dataset storage
├── logs/                  # Training logs
├── checkpoints/           # Model checkpoints
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- albumentations 1.3+
- numpy 1.24+
- pyyaml 5.4+
- torchmetrics 1.0+

## Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The CustomCNN architecture features:
- Input → C1 (RF: 3) → C2 (RF: 5) → Depthwise Sep (RF: 7) →
- Dilated (RF: 15) → C3 with stride 2 (RF: 31) → C4 with stride 2 (RF: 47) →
- GAP → FC → Output

Key characteristics:
- No MaxPooling (uses strided convolutions)
- Uses Depthwise Separable Convolution for efficiency
- Uses Dilated Convolution for expanded receptive field
- Under 200k parameters
- Batch Normalization after each convolution
- ReLU activation throughout

## Data Augmentation

The following augmentations are applied using Albumentations:
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(
        max_holes=1, max_height=16, max_width=16,
        min_holes=1, min_height=16, min_width=16,
        fill_value=mean.tolist(), mask_fill_value=None, p=0.5
    ),
    ToTensorV2()
])
```

## Training

The training process includes:
- Dynamic progress bars for:
  - Training progress
  - Evaluation progress
  - Accuracy progress towards target
- Comprehensive metrics using torchmetrics:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
- Automatic checkpoint saving for best models
- Target accuracy tracking (85%)

## Usage

1. Train the model:
```bash
python scripts/train.py
```

2. Evaluate the model:
```bash
python scripts/evaluate.py
```

## Configuration

Modify `config/config.yaml` to adjust settings:

```yaml
training:
  batch_size: 64
  num_epochs: 20
  learning_rate: 0.001
  save_freq: 5
  print_freq: 100

model:
  num_classes: 10
  checkpoint_file: 'cifar10_model.pth'

paths:
  checkpoint_dir: 'checkpoints'
  data_dir: 'data'
  logs_dir: 'logs'

metrics:
  target_accuracy: 85.0
```
