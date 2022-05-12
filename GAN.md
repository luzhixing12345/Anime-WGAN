# GAN

GAN的基本原理其实非常简单.假设我们有两个网络,G(Generator)和D(Discriminator)

- G是一个生成图片的网络,它接收一个随机的噪声z,通过这个噪声生成图片,记做G(z)
- D是一个判别网络,判别一张图片是不是"真实的". 它的输入参数是x,x代表一张图片,输出D(x)代表x为真实图片的概率. 0 ~ 1 的范围表示判别器认为这张图像是真实图像的概率是多少.

在训练过程中,生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D.而D的目标就是尽量把G生成的图片和真实的图片分别开来.这样,G和D构成了一个动态的"博弈过程".

最后博弈的结果是什么? 在最理想的状态下,G可以生成足以"以假乱真"的图片G(z).对于D来说,它难以判定G生成的图片究竟是不是真实的,因此D(G(z)) = 0.5.

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
  H_{out} = \frac{H_{in}+2\times padding[0]-dilation[0]\times (kernel_size[0]-1)-1}{stride[0]}+1 = \frac{96+2\times 1-1\times (4-1)-1}{2}+1 = 48
  $$
  $$
  W_{out} = \frac{W_{in}+2\times padding[0]-dilation[0]\times (kernel_size[0]-1)-1}{stride[0]}+1 = \frac{96+2\times 1-1\times (4-1)-1}{2}+1 = 48
  $$

  $$
  kernel_size = 4 \And stride = 2 \And padding = 1 \And deliation = 1 \rightleftharpoons H_out = \frac{H_in}{2}
  $$

- [nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html?highlight=nn%20convtranspose2d#torch.nn.ConvTranspose2d)

  $$
  H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
  $$
  
- [nn.PixelShuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html?highlight=nn%20pixelshuffle#torch.nn.PixelShuffle)

## Structure

```txt
          module name   input shape  output shape      params memory(MB)             MAdd        Flops  MemRead(B)  MemWrite(B) duration[%]   MemR+W(B)
0       main_module.0   100   1   1  1024   4   4   1638400.0       0.06      3,275,776.0          0.0         0.0          0.0       9.50%         0.0
1       main_module.1  1024   4   4  1024   4   4      2048.0       0.06         65,536.0     32,768.0     73728.0      65536.0       3.18%    139264.0
2       main_module.2  1024   4   4  1024   4   4         0.0       0.06         16,384.0     16,384.0     65536.0      65536.0       0.00%    131072.0
3       main_module.3  1024   4   4   512   8   8   8388608.0       0.12    268,427,264.0          0.0         0.0          0.0      15.90%         0.0
4       main_module.4   512   8   8   512   8   8      1024.0       0.12        131,072.0     65,536.0    135168.0     131072.0       3.17%    266240.0
5       main_module.5   512   8   8   512   8   8         0.0       0.12         32,768.0     32,768.0    131072.0     131072.0       0.00%    262144.0
6       main_module.6   512   8   8   256  16  16   2097152.0       0.25    268,419,072.0          0.0         0.0          0.0       6.36%         0.0
7       main_module.7   256  16  16   256  16  16       512.0       0.25        262,144.0    131,072.0    264192.0     262144.0       0.00%    526336.0
8       main_module.8   256  16  16   256  16  16         0.0       0.25         65,536.0     65,536.0    262144.0     262144.0       0.00%    524288.0
9       main_module.9   256  16  16   128  32  32    524288.0       0.50    268,402,688.0          0.0         0.0          0.0       9.64%         0.0
10     main_module.10   128  32  32   128  32  32       256.0       0.50        524,288.0    262,144.0    525312.0     524288.0       3.18%   1049600.0
11     main_module.11   128  32  32   128  32  32         0.0       0.50        131,072.0    131,072.0    524288.0     524288.0       0.00%   1048576.0
12     main_module.12   128  32  32    64  64  64    131072.0       1.00    268,369,920.0          0.0         0.0          0.0      12.71%         0.0
13     main_module.13    64  64  64    64  64  64       128.0       1.00      1,048,576.0    524,288.0   1049088.0    1048576.0       0.00%   2097664.0
14     main_module.14    64  64  64    64  64  64         0.0       1.00        262,144.0    262,144.0   1048576.0    1048576.0       0.00%   2097152.0
15     main_module.15    64  64  64    32 128 128     32768.0       2.00    268,304,384.0          0.0         0.0          0.0      19.28%         0.0
16     main_module.16    32 128 128    32 128 128        64.0       2.00      2,097,152.0  1,048,576.0   2097408.0    2097152.0       0.00%   4194560.0
17     main_module.17    32 128 128    32 128 128         0.0       2.00        524,288.0    524,288.0   2097152.0    2097152.0       0.00%   4194304.0
18     main_module.18    32 128 128     3 256 256      1536.0       0.75     50,282,496.0          0.0         0.0          0.0      13.16%         0.0
19             output     3 256 256     3 256 256         0.0       0.75              0.0          0.0         0.0          0.0       3.91%         0.0
total                                              12817856.0      13.31  1,400,642,560.0  3,096,576.0         0.0          0.0     100.00%  16531200.0
=======================================================================================================================================================
Total params: 12,817,856
-------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 13.31MB
Total MAdd: 1.4GMAdd
Total Flops: 3.1MFlops
Total MemR+W: 15.77MB
```