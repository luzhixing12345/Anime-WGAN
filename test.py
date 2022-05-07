
import torch
import torch.nn as nn


channels = 3
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Input_dim = channels (CxHxW)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # State (CxHxW)
            # default (32, 3, 64, 64)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x H/2 x W/2)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x H/4 x W/4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x H/8 x W/8)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())
        
            # output size (1 x (H/8-3) x (W/8-3))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    
z = torch.rand((32, 3, 64, 64))

s = Discriminator(channels)(z)
print(s.shape)
