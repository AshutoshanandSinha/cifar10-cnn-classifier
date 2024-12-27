import torchvision
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

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, label

    def __len__(self):
        return len(self.dataset)

def get_transforms(mean, std, is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=[x * 255 for x in mean], p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

def load_data():
    # Calculate mean and std
    initial_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True
    )
    
    mean = np.zeros(3)
    std = np.zeros(3)
    for img, _ in initial_dataset:
        img = np.array(img) / 255.0
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
    
    mean /= len(initial_dataset)
    std /= len(initial_dataset)
    
    print("Calculated Mean:", mean)
    print("Calculated Std:", std)

    # Create transforms with proper normalization
    train_transform = get_transforms(mean, std, is_train=True)
    test_transform = get_transforms(mean, std, is_train=False)

    # Create datasets
    train_dataset_base = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True
    )
    test_dataset_base = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True
    )

    # Wrap with Albumentations
    train_dataset = AlbumentationsDataset(train_dataset_base, transform=train_transform)
    test_dataset = AlbumentationsDataset(test_dataset_base, transform=test_transform)

    # Create data loaders with proper batch size from config
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,  # Use uppercase config attributes
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,  # Use uppercase config attributes
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader
