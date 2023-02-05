import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gap = nn.Sequential (
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dw_conv(x)
        x = self.dilated_conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



