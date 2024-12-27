import torch.nn as nn
from ..utils.config import config

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 - Initial convolution block with more channels, Receptive Field: 3
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        
        # C2 - Increase channels gradually, Receptive Field: 5
        self.c2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1)
        )  #RF: 5

        # C3 conv with stride 2, Receptive Field: 7
        self.c3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )  #RF: 7

        # C4 - conv with dilation, Receptive Field: 11  
        self.c4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )  #RF: 11

        # C5 - depthwise separable conv, Receptive Field: 23
        self.c5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1, bias=False),  # Depthwise
            nn.Conv2d(32, 48, kernel_size=1, bias=False),  # Pointwise, reduced from 128
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1)
        )  #RF: 23

        # C6 - conv with stride 2, Receptive Field: 27
        self.c6 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=0, bias=False),  # Reduced from 128
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1)
        )  #RF: 27

        # C7 - conv 
        self.c7 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(0.1)
        )  #RF: 31  

        # C8 - conv, Receptive Field: 35
        self.c8 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )  #RF: 35
        

        # GAP and FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(48, config.NUM_CLASSES)  # Changed from 96 to 48 to match the channel size

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten properly preserving batch size
        x = self.fc(x)
        return x

