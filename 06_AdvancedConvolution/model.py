import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.dw_conv = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64)
        self.dilated_conv = nn.Conv2d(64, 64, 3, stride=1, dilation=2, padding=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.dw_conv(x))
        x = F.relu(self.dilated_conv(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
