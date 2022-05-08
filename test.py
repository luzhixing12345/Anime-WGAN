
import torch
import torch.nn as nn


model = Generator(channels=3)

a = torch.randn(32, 100, 1, 1)

b = model(a)

print(b.shape)