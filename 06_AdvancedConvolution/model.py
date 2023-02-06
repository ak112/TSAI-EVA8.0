import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(3, 128, 5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),

            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            
            nn.Conv2d(32, 32, 3, dilation=2, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv3 = nn.Sequential(
            
            nn.Conv2d(32, 32, 3, stride=2, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 32, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )

        self.conv4 = nn.Sequential(
            
            nn.Conv2d(32, 32, 3, stride=2, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 10, 3, stride=1, padding=1, bias=False),

        )


        self.gap = nn.Sequential(
            
            nn.AvgPool2d(kernel_size=3),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)



