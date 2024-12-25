import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from ..utils.config import config

class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)  # Convert PIL image to numpy array

        # Convert to float and normalize to [0, 1] range
        img = img.astype(np.float32) / 255.0

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, label

    def __len__(self):
        return len(self.dataset)

def get_transforms(mean):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16,
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=mean.tolist(), mask_fill_value=None, p=0.5),
        ToTensorV2()
    ])

def load_data():
    # Initial dataset to calculate mean and std
    initial_dataset = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True)

    # Calculate mean and std on the fly
    mean = np.zeros(3)
    std = np.zeros(3)
    total_images = 0

    for img, _ in initial_dataset:
        img = np.array(img) / 255.0
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
        total_images += 1

    mean /= total_images
    std /= total_images

    print("Calculated Mean:", mean)
    print("Calculated Std:", std)

    transform = get_transforms(mean)

    # Create base datasets
    train_dataset_base = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True)
    test_dataset_base = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=False, download=True)

    # Wrap with Albumentations dataset
    train_dataset = AlbumentationsDataset(train_dataset_base, transform=transform)
    test_dataset = AlbumentationsDataset(test_dataset_base, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader
