# Anime-WGAN-GP

## Inroduction: Using WGAN-GP to generate anime faces

![WGAN64_anime](https://raw.githubusercontent.com/learner-lu/picbed/master/WGAN64_anime.png)

TODO: an online website

[Bilibili video](123)

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

  I crawled pictures with high resolution and manually removed some unsatisfied pictures. The whole dataset contains 2.7w+ anime faces with resolution of 256x256 in high quality

- [anime](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.2/faces.zip)

  Another dataset I found, 5w+, resolution of 96x96, in good quality

download the dataset zip and unzip it under `dataset` as `dataset/anime256` and `dataset/anime`

> try to use [Github proxy](https://ghproxy.com/) if too slow
>
> see [more information about anime datasets](https://github.com/luzhixing12345/anime-face-dataset)

## Use

### Train

I design two model structures.

- run WGAN model as

  ```bash
  python train.py --config-file configs/WGAN.yaml
  ```

- run WGANP model as

  ```bash
  python train.py --config-file configs/WGANP.yaml
  ```

**ATTENTION**: Actually this project is a bit complex than its introduction, if you really want to run this code and train a model by yourself, see more information in [train-info.md](./train-info.md)

### Generate an anime picture

- use WGAN model as

  ```bash
  python generate.py --config-file configs/WGAN.yaml -g checkpoints/WGAN/WGAN_G_epoch_39999.pth.pth
  ```

- use WGANP model as

  ```bash
  python generate.py --config-file configs/WGANP.yaml -g checkpoints/WGANP/WGANP_G_epoch_39999.pth.pth
  ```

by default it will generate an 8x8 grid fake anime image under `./images`

Arguments:

- `-g`: short for generator, one argument with the path name of the model weights
- `-s`: short for separate, no argument. Use `-s` to separate the grid image into 64 images

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

see more information about [Supercomputing Center of Wuhan University](http://hpc.whu.edu.cn/index.htm)

|model|dataset|Discriminator|Generator|
|:--:|:--:|:--:|:--:|
|WGAN|ANIME256|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_D_ANIME256.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_G_ANIME256.pth)|
||ANIME|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_D_ANIME.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_G_ANIME.pth)|
|WGANP|ANIME256|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_D_ANIME256.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_G_ANIME256.pth)|
||ANIME|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_D_ANIME.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_G_ANIME.pth)|

If you don't want to train by yourself(about 36~48h), download the pretrained model G and move it under `./checkpoints`, then run generate.py as above to generate images, **remember to use the correct path name.**

> actually if you just want to generate an anime picture you only need to download G, but whatever, I uploaded all.

## Detail information about this project

If you are familiar with **Chinese**, you could visit [my blog](https://luzhixing12345.github.io/tags/GAN/) which recorded my understanding of GAN, DCGAN, WGAN-CP, WGAN-GP, the evaluation method of GAN, and the whole project.

Sorry for my poor english, translate it into english if you really need it.

## Result

- generated fake images

  > Actually not all generated images seem well, I manually choose some images I like from `WGAN + anime256`

  ![7](https://raw.githubusercontent.com/learner-lu/picbed/master/7.png) ![33](https://raw.githubusercontent.com/learner-lu/picbed/master/33.png) ![61](https://raw.githubusercontent.com/learner-lu/picbed/master/61.png) ![18](https://raw.githubusercontent.com/learner-lu/picbed/master/18.png) ![12](https://raw.githubusercontent.com/learner-lu/picbed/master/12.png) ![13](https://raw.githubusercontent.com/learner-lu/picbed/master/13.png)

  well, cute girls, doesn't it?

- walking latent space

  > view my blog if you are not familiar with latent space

  |WGAN + anime256|WGAN + anime|
  |:--:|:--:|
  |![1](https://raw.githubusercontent.com/learner-lu/picbed/master/walking_latent_space.gif)|![asdoqi](https://raw.githubusercontent.com/learner-lu/picbed/master/asdoqi.gif)|

  |WGANP + anime256|WGANP + anime|
  |:--:|:--:|
  |![asdhjono](https://raw.githubusercontent.com/learner-lu/picbed/master/asdhjono.gif)|![hcouga](https://raw.githubusercontent.com/learner-lu/picbed/master/hcouga.gif)|

- GAN training process

  > for the same noise input, the process of different generated images

  |WGAN + anime256|WGAN + anime|
  |:--:|:--:|
  |![ANIME256_process](https://raw.githubusercontent.com/learner-lu/picbed/master/ANIME256_process.gif)|![WGAN64_anime_process](https://raw.githubusercontent.com/learner-lu/picbed/master/WGAN64_anime_process.gif)|

  |WGANP + anime256|WGANP + anime|
  |:--:|:--:|
  |![ANIME256P_process](https://raw.githubusercontent.com/learner-lu/picbed/master/ANIME256P_process.gif)|![WGAN64P_anime_process](https://raw.githubusercontent.com/learner-lu/picbed/master/WGAN64P_anime_process.gif)|

## Conclusion

This is my first try of GAN, I have heard of it many times but never try to learn it. Concidentally here comes my final report of [data science introduction](https://github.com/luzhixing12345/data-science-introduction) lesson, and the final homework is to do something related to data. So that's a good chance to learn GAN ! I read some papers of GAN and it's really interesting.

[An article](https://zhuanlan.zhihu.com/p/24767059) from Zhihu inspired me that why not try to generate my own anime faces/avatar? So I start to code.

Actually I must admit that my pretrained model doesn't perform as well as I excepted, there maybe many excellent model structures or projects which can solve this problem well today. However, the origin purposal of this project is just to learn something about GAN, I choose to use WGAN-GP for it's wonderfull mathematical derivation, I'd like to finish my own project instead of just git clone an excellent project and run without doing anything.

Especialy thank to [pytorch-wgan](https://github.com/Zeleni9/pytorch-wgan), really helps me a lot. Thank you so much.

Actually at first I want to generate 256x256 images, that's why I crawled 256x256 images and made that dataset. But as the size of image multiply 4, the model performs extremly bad. Maybe I need a better model struture. I've tried to use residual block but it doesn't work well.

If you have any other question, leave your confusion in Issue and I will apply as soon as possible.

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
- https://github.com/bchao1/Anime-Face-Dataset
- https://github.com/jayleicn/animeGAN
- [Generative adversial nets](https://arxiv.org/abs/1406.2661) was published in 2014 NIPSd
- [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) which using CNN in GAN has made a huge improvement.
- https://github.com/nagadomi/lbpcascade_animeface
- https://github.com/xiaoyou-bilibili/anime_avatar_gen
- https://zhuanlan.zhihu.com/p/25071913
- https://zhuanlan.zhihu.com/p/58260684
- https://arxiv.org/pdf/1708.05509.pdf
- [qipan](https://distill.pub/2016/deconv-checkerboard/)
- https://www.zhihu.com/search?type=content&q=GAN%20%E6%A3%8B%E7%9B%98
- https://zhuanlan.zhihu.com/p/58260684
- https://make.girls.moe/#/
- [latent walk](https://www.zhihu.com/search?type=content&q=latent%20walk)