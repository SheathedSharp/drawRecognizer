import torch
import torch.nn as nn


class DigitCNN_C16C32_K3_FC10(nn.Module):
    """
    2层CNN架构 (16->32通道, 3x3卷积核)
    - Conv1: 1->16 channels, 3x3 kernel
    - Conv2: 16->32 channels, 3x3 kernel
    - FC: 32*7*7->64->10
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DigitCNN_C32C64_K5_FC512(nn.Module):
    """
    2层CNN架构 (32->64通道, 5x5卷积核)
    - Conv1: 1->32 channels, 5x5 kernel
    - Conv2: 32->64 channels, 5x5 kernel
    - FC: 64*7*7->512->10
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DigitCNN_C32C64C128_K3_FC256(nn.Module):
    """
    3层CNN架构 (32->64->128通道, 3x3卷积核)
    - Conv1: 1->32 channels, 3x3 kernel
    - Conv2: 32->64 channels, 3x3 kernel
    - Conv3: 64->128 channels, 3x3 kernel
    - FC: 128*3*3->256->10
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DigitMLP_512_256_128(nn.Module):
    """
    3层MLP架构
    - FC1: 784->512
    - FC2: 512->256
    - FC3: 256->128
    - FC4: 128->10
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
