# Anime-WGAN-GP

[english README](en-README.md)

[Live Demo](https://visual.kamilu.top) (more [info](https://github.com/luzhixing12345/pytorch-model-deployment/tree/web-server))

[Bilibili video](https://www.bilibili.com/video/BV1cr4y147s8)

## Introduction: Using WGAN-GP to generate anime faces

![WGAN64_anime](https://raw.githubusercontent.com/learner-lu/picbed/master/WGAN64_anime.png)

## Requirements

- python : 3.7
- [pytorch](https://pytorch.org/get-started/previous-versions/)

  ```bash
  # CUDA 10.2
  conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
  ```

- other dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Dataset

- [anime256](https://github.com/luzhixing12345/anime-face-dataset/releases/download/v0.0.1/anime256.zip)

  整个数据集包含 2.7w+ 分辨率为 256x256 的高质量动漫人脸

  > https://github.com/luzhixing12345/anime-face-dataset

- [anime](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.2/faces.zip)

  另一个数据集，5w+，分辨率96x96，质量不错

下载zip并且解压到 `./dataset` 文件夹下, `dataset/anime256` and `dataset/anime`

> 如果下载过慢可以使用[Github proxy](https://ghproxy.com/)加速
>
> 这两个数据集下载任意一个用于训练都可以

## Use

### Train

我设计了两个模型结构, `WGAN` 和 `WGANP`, 这两个模型最主要的区别在于Generator增大图片尺寸的方式,你可以通过这篇文章[反卷积存在的问题](https://distill.pub/2016/deconv-checkerboard/)了解到为什么要这么做

- run WGAN model as

  ```bash
  python train.py --config-file configs/WGAN.yaml
  ```

- run WGANP model as

  ```bash
  python train.py --config-file configs/WGANP.yaml
  ```

**注意**: 如果需要自己训练模型,请参阅[更多训练信息](./train-info.md),当然你可以在后面找到提供的预训练模型

### Generate images

- use WGAN model as

  ```bash
  python generate.py --config-file configs/WGAN.yaml -g checkpoints/WGAN/WGAN_G_epoch_39999.pth
  ```

- use WGANP model as

  ```bash
  python generate.py --config-file configs/WGANP.yaml -g checkpoints/WGANP/WGANP_G_epoch_39999.pth
  ```

- use CPU

  ```bash
  python generate.py --config-file configs/WGANP.yaml -g checkpoints/WGANP_G_ANIME256.pth MODEL.DEVICE cpu
  ```

默认情况下,它会在 `./images` 下生成一个 8x8 网格的动漫图片

其他:

- `-g`: generator 的缩写,一个参数是模型权重的路径名

  **这里的模型是指生成器G的模型**

- `-s`: separate 的缩写,没有参数

  可以使用`-s`将将一张大图拆分成每一张小图

## Pretrained model

- software

  ```txt
  OS: CentOS 7.5 Linux X86_64
  Python: 3.7.14 (anaconda)
  PyTorch: 1.10.1
  ```

- hardware

  ```txt
  CPU: Intel Xeon 6226R
  GPU: Nvidia Tesla V100 16GB
  ```

|model|dataset|Discriminator|Generator|
|:--:|:--:|:--:|:--:|
|WGAN|ANIME256|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_D_ANIME256.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_G_ANIME256.pth)|
||ANIME|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_D_ANIME.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_G_ANIME.pth)|
|WGANP|ANIME256|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_D_ANIME256.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_G_ANIME256.pth)|
||ANIME|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_D_ANIME.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_G_ANIME.pth)|

如果不想自己训练（大约36~48h）,可以下载预训练好的模型G并将其移动到`./checkpoints`下,然后按照上文`Generate images`生成图片, **注意修改后面的生成器G的模型路径**

> 事实上如果你只是想生成一张动漫图片你只需要下载G, 判别器D在生成图片的过程中并没有用到

## Detail information about this project

我在[博客](https://luzhixing12345.github.io/tags/GAN/)中记录了我对GAN、DCGAN、WGAN-CP、WGAN-GP的理解，GAN的评估方法,如果对GAN尚不足够了解可以参考这一部分

## Result

- generated fake images

  > 其实不是所有生成的图片都好看,我手动选择了一些我喜欢的图片,使用模型 `WGAN + anime256`

  ![7](https://raw.githubusercontent.com/learner-lu/picbed/master/7.png) ![33](https://raw.githubusercontent.com/learner-lu/picbed/master/33.png) ![61](https://raw.githubusercontent.com/learner-lu/picbed/master/61.png) ![18](https://raw.githubusercontent.com/learner-lu/picbed/master/18.png) ![12](https://raw.githubusercontent.com/learner-lu/picbed/master/12.png) ![13](https://raw.githubusercontent.com/learner-lu/picbed/master/13.png)

- walking latent space

  > 如果尚不了解潜在空间探索可以参考[博客](https://luzhixing12345.github.io/2022/05/18/GAN/GAN%E7%BD%91%E7%BB%9C%E8%AF%A6%E8%A7%A3(%E4%BA%8C)/)

  |WGAN + anime256|WGAN + anime|
  |:--:|:--:|
  |![1](https://raw.githubusercontent.com/learner-lu/picbed/master/walking_latent_space.gif)|![asdoqi](https://raw.githubusercontent.com/learner-lu/picbed/master/asdoqi.gif)|

  |WGANP + anime256|WGANP + anime|
  |:--:|:--:|
  |![asdhjono](https://raw.githubusercontent.com/learner-lu/picbed/master/asdhjono.gif)|![hcouga](https://raw.githubusercontent.com/learner-lu/picbed/master/hcouga.gif)|

- GAN training process

  > 对于相同的噪声输入,不同生成图像的过程

  |WGAN + anime256|WGAN + anime|
  |:--:|:--:|
  |![ANIME256_process](https://raw.githubusercontent.com/learner-lu/picbed/master/ANIME256_process.gif)|![WGAN64_anime_process](https://raw.githubusercontent.com/learner-lu/picbed/master/WGAN64_anime_process.gif)|

  |WGANP + anime256|WGANP + anime|
  |:--:|:--:|
  |![ANIME256P_process](https://raw.githubusercontent.com/learner-lu/picbed/master/ANIME256P_process.gif)|![WGAN64P_anime_process](https://raw.githubusercontent.com/learner-lu/picbed/master/WGAN64P_anime_process.gif)|

## Conclusion

这是我第一次尝试GAN,久闻大名但从未尝试学习它。恰逢[数据科学导论](https://github.com/luzhixing12345/data-science-introduction)课期末报告，期末作业是做一些与机器学习相关的事情。所以这是学习 GAN 的好机会！于是看了一些GAN的论文，真的很有意思。

[GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)启发了尝试生成自己的动漫头像

实际上我必须承认,我的预训练模型并没有像我期待的那样表现出色,时至今日有很多优秀的模型在生成图像上具有更好的效果,比如diffusion.

这个项目的最初目的只是为了学习一些关于 GAN 的东西，我选择使用 WGAN-GP 是因为它的数学推导很棒，我想尝试动手写一下而不是只是双击运行,那样稍显无趣.

特别感谢 [pytorch-wgan](https://github.com/Zeleni9/pytorch-wgan),绝大部分代码都参考自这个项目

其实一开始我想生成 256x256 的图像，这就是为什么我爬取 256x256 图像并制作该数据集的原因。但随着图像尺寸乘以 4，模型表现极差。也许我需要一个更好的模型结构。我试过使用residual block ,但效果不佳。

## Relevant reference

project:

- [Github-GAN+DCGAN+WGAN-pytorch](https://github.com/Zeleni9/pytorch-wgan) (easy to use,recommand)
- [Github-GAN*-pytorch](https://github.com/eriklindernoren/PyTorch-GAN) (overall)
- [Github-DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [Github-GAN*-tensorflow](https://github.com/YadiraF/GAN)

knowledge:

- [知乎-GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)
- [GAN video by Li mu](https://www.bilibili.com/video/BV1rb4y187vD)
- [KL散度](https://zhuanlan.zhihu.com/p/365400000)
- [GAN](https://www.zhihu.com/search?q=GAN&type=content&sort=upvoted_count)
- [WGAN-DCGAN](https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py)
- [WGAN-GP](https://github.com/EmilienDupont/wgan-gp)
- https://zhuanlan.zhihu.com/p/28407948?ivk_sa=1024320u
- [evaluation-index](https://zhuanlan.zhihu.com/p/432965561)
- https://zhuanlan.zhihu.com/p/25071913
- https://zhuanlan.zhihu.com/p/58260684
- [生成ANIMEfaces](https://arxiv.org/pdf/1708.05509.pdf)
- [checkerboard](https://distill.pub/2016/deconv-checkerboard/)
- https://www.zhihu.com/search?type=content&q=GAN%20%E6%A3%8B%E7%9B%98
- https://zhuanlan.zhihu.com/p/58260684
- https://make.girls.moe/#/
- [latent walk](https://www.zhihu.com/search?type=content&q=latent%20walk)
