# CIFAR-10 Image Classifier

This project implements a custom CNN architecture for classifying images from the CIFAR-10 dataset using PyTorch.

## Features

- Custom CNN architecture with:
  - Depthwise Separable Convolution
  - Dilated Convolution
  - Strided Convolutions (instead of MaxPooling)
  - Global Average Pooling
  - Total Receptive Field > 35
  - Under 100k parameters (~85.6k)
- Data augmentation using Albumentations
- On-the-fly mean and standard deviation calculation
- Checkpoint management and model saving
- Training progress visualization with:
  - Live progress bars
  - Training history plots
  - Detailed metrics logging
- YAML-based configuration
- GPU support with automatic device selection

## Project Structure

```
cifar10_project/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── transforms.py
│   ├── models/            # Model definitions
│   │   ├── __init__.py
│   │   └── cnn.py
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── checkpoints.py
│   │   ├── setup.py
│   │   ├── visualization.py
│   │   └── model_analysis.py
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
├── checkpoints/          # Model checkpoints
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Requirements

```
numpy==1.24.3
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
pyyaml>=5.4.1
torchmetrics>=1.0.0
torchsummary>=1.5.1
```

## Setup

1. Create a virtual environment:
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
- Input (3x32x32) → C1 (16) → C2 (24) → C3 (32, stride=2) →
- C4 (32, dilated) → C5 (48, depthwise sep) → C6 (48, stride=2) →
- C7 (48) → C8 (48) → GAP → FC (10)

Key characteristics:
- Uses strided convolutions instead of MaxPooling
- Employs Depthwise Separable Convolution for efficiency
- Uses Dilated Convolution for expanded receptive field
- ~85.6k parameters
- Batch Normalization and Dropout after each convolution
- ReLU activation throughout

## Data Augmentation

Using Albumentations with:
- Random Crop (32x32)
- Horizontal Flip (p=0.5)
- Shift-Scale-Rotate
- Coarse Dropout
- Color Jitter
- Normalization

## Training

Features:
- OneCycleLR scheduler
- Live progress tracking with tqdm
- Comprehensive metrics:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
- Automatic checkpoint saving
- Training history visualization
- Target accuracy tracking (85%)

## Configuration

Key settings in `config/config.yaml`:

```yaml
training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.01
  weight_decay: 1e-4
  save_freq: 5
  print_freq: 100

model:
  num_classes: 10
  checkpoint_file: 'cifar10_model.pth'
  dropout_rate: 0.1

metrics:
  target_accuracy: 85.0
```

## Usage

1. Train the model:
```bash
python scripts/train.py
```

2. Evaluate the model:
```bash
python scripts/evaluate.py
```

## Model Analysis

The model provides detailed analysis including:
- Layer-by-layer parameter counts
- Memory usage statistics
- Forward/backward pass size
- Total parameter count (~85.6k)
- Model size in MB (0.33 MB)

## Training Results

### Model Performance
- **Target Accuracy**: 85.0%
- **Best Accuracy Achieved**: 85.57%
- **Final Model Metrics**:
  - Accuracy: 85.49%
  - Loss: 0.4239
  - F1 Score: 0.8549
  - Precision: 0.8549
  - Recall: 0.8549

### Training Progress
The model showed steady improvement throughout training:
- Reached 70% accuracy by epoch 5
- Crossed 75% accuracy at epoch 11
- Achieved 80% accuracy by epoch 23
- Exceeded target accuracy of 85% at epoch 42
- Maintained consistent performance above 85% for final epochs

### Model Architecture
- Total Parameters: 85,034
- Model Size: 0.32 MB
- Architecture: Custom CNN with 8 convolutional blocks
- Final Layer: Global Average Pooling followed by Dense layer

### Training Plots
Training history plots are saved in `logs/training_history.png`
