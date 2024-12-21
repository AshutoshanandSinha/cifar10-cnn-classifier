import torch.nn as nn
from ..utils.config import config

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # C1 Block - RF: 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # C2 Block - RF: 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Depthwise Separable Conv Block - RF: 7
        self.depthwise = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(48)
        )

        # Dilated Conv Block - RF: 15
        self.dilated = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(48)
        )

        # C3 Block with stride 2 - RF: 31
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )

        # C4 Block with stride 2 - RF: 47
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, config.NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.depthwise(x)
        x = self.dilated(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def count_parameters(model):
    """
    Calculate parameters for each layer
    """
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    return total_params

# Parameter calculations:
# Conv1: (3 * 3 * 3 * 16) + 16 = 448 params
# BN1: 2 * 16 = 32 params

# Conv2: (3 * 3 * 16 * 32) + 32 = 4,640 params
# BN2: 2 * 32 = 64 params

# Depthwise: (3 * 3 * 32) + 32 = 320 params
# BN_depth: 2 * 32 = 64 params
# Pointwise: (1 * 1 * 32 * 48) + 48 = 2,112 params
# BN_point: 2 * 48 = 96 params

# Dilated: (3 * 3 * 48 * 48) + 48 = 36,928 params
# BN_dilated: 2 * 48 = 96 params

# Conv3: (3 * 3 * 48 * 96) + 96 = 73,856 params
# BN3: 2 * 96 = 192 params

# Conv4: (3 * 3 * 96 * 128) + 128 = 295,168 params
# BN4: 2 * 128 = 256 params

# FC: (128 * 10) + 10 = 1,290 params
