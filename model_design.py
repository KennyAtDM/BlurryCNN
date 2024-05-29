from torch import nn
import torch

class BCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,kernel_size=3,stride=2),
            torch.nn.BatchNorm2d(32),
        )
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=5),
            torch.nn.BatchNorm2d(64),
        )
        self.act2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,kernel_size=5,stride=1),
            torch.nn.BatchNorm2d(128),
        )
        self.act3 = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.head = torch.nn.Sequential(
                        torch.nn.Flatten(),
                        torch.nn.Dropout(p=0.3, inplace=True), 
                        torch.nn.Linear(in_features=128, # checked by running without head
                                        out_features=2, # same number of output units as our number of classes
                                        bias=True))
    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.avgpool(x)
        x = self.head(x)
        return x