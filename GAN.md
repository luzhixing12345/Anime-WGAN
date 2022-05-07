# GAN

![20220506135915](https://raw.githubusercontent.com/learner-lu/picbed/master/20220506135915.png)

![20220506135928](https://raw.githubusercontent.com/learner-lu/picbed/master/20220506135928.png)

![gan](https://raw.githubusercontent.com/learner-lu/picbed/master/gan.png)

## pytorch

- [nn.Conv2d()](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=nn%20conv2d#torch.nn.Conv2d)

  > [CSDN-blog](https://blog.csdn.net/qq_42079689/article/details/102642610)

  nn.Conv2d(**in_channels, out_channels, kernel_size, stride=1, padding=0**)

  ```python
  import torch
  import torch.nn as nn

  input_tensor = torch.rand((32, 3, 96, 96))
  layer = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1)

  # output
  output_tensor = layer(input_tensor)
  print(output_tensor.shape())  # (32,256,48,48)
  ```

  $$
  H_{out} = \frac{H_{in}+2\times padding[0]-dilation[0]\times (kernel_size[0]-1)}{stride[0]}+1 = \frac{96-2\times 1-1\times (4-1)}{2}+1 = 48
  $$
  $$
  W_{out} = \frac{W_{in}+2\times padding[0]-dilation[0]\times (kernel_size[0]-1)}{stride[0]}+1 = \frac{96-2\times 1-1\times (4-1)}{2}+1 = 48
  $$

  $$
  kernel_size = 4 \And stride = 2 \And padding = 1 \And deliation = 1 \rightleftharpoons H_out = \frac{H_in}{2}
  $$

- [nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html?highlight=nn%20convtranspose2d#torch.nn.ConvTranspose2d)

  $$
  H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
  $$
  