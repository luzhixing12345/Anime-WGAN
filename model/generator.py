'''
*Copyright (c) 2022 All rights reserved
*@description: generator for model-256
*@author: Zhixing Lu
*@date: 2022-05-12
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels = 3, dimension = 1024, input_size = 100):
        super().__init__()
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            
            # input is Z which size is (batch size x C x 1 X 1),going into a convolution
            # by default (bs, 100, 1, 1)
            # project and reshape
            nn.ConvTranspose2d(in_channels=input_size, out_channels=dimension, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=dimension),
            nn.ReLU(True),
            
            # LAYER1
            # State (bs, 1024 x 4 x 4)
            Block(dimension//4),

            # LAYER2
            # State (bs, 256 x 8 x 8)
            Block(dimension//16),
            
            # LAYER3
            # State (bs, 64 x 16 x 16)
            Block(dimension//64),
            
            # LAYER4
            # State (bs, 16 x 32 x 32)
            nn.ConvTranspose2d(in_channels=dimension//64, out_channels=dimension//64*3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=dimension//64*3),
            nn.ReLU(True),
            
            # LAYER5
            # State (bs, 48 x 64 x 64)
            Block(dimension//64//16*3, 4),
            )
            # output of main module --> Image (batch size x C x 64 x 64)
            # default (32, 3, 256, 256)

        # output activation function is Tanh
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Block(nn.Module):
    
    def __init__(self, dimension, r = 2,bias=False):
        
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels=dimension, out_channels=dimension*2, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(dimension*2)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=dimension*2, out_channels=dimension, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dimension)

    
    def forward(self, tensor):
        
        output = self.pixel_shuffle(tensor)
        residual_output = output
        output = self.upsample(output)
        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += residual_output
        return output
        