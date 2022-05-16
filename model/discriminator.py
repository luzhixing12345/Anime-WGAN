'''
*Copyright (c) 2022 All rights reserved
*@description: discriminator for model-256
*@author: Zhixing Lu
*@date: 2022-05-12
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels = 3, dimension = 256):
        super().__init__()
        # Input_dim = channels (CxHxW)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # State (CxHxW)
            # default (32, 3, 256, 256)
            nn.Conv2d(in_channels=channels, out_channels=dimension, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=dimension),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x 128 x 128)
            Block(dimension),
            
            # State (256x 32 x 32)
            Block(dimension//2),
            
            # State (256x 8 x 8)
            Block(dimension//4),
            
            # State (256 x 2 x 2)
            )
        
        
            # outptut of main module --> State (8192x 4 x 4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=dimension//8, out_channels=1, kernel_size=4, stride=2, padding=1))
            # remove sigmoid function
            # output size (1 x (H/16-3) x (W/16-3))
            # default (32, 1, 1, 1)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Block(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dimension, dimension//2, kernel_size = 4, stride = 4, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(dimension//2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(dimension//2, dimension//2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(dimension//2)
        self.conv3 = nn.Conv2d(dimension//2, dimension//2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        r_x = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = x + r_x
        x = self.relu3(x)
        return x
    