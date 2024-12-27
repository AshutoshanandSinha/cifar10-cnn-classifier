import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mean, std):
    train_transform = A.Compose([
        A.RandomCrop(32, 32),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=1, max_height=8, max_width=8, 
            min_holes=1, min_height=8, min_width=8, 
            fill_value=mean, p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, test_transform 