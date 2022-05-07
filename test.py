
import torch
from PIL import Image
import numpy as np

a = torch.randn(1, 3, 64, 64)
a = a.view((3*8,8,64))
print(a.shape)



