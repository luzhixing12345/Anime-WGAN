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

I design two model structure.

- run WGAN model as

  ```bash
  python train.py --config-file configs/WGAN.yaml
  ```

- run WGANP model as

  ```bash
  python train.py --config-file configs/WGANP.yaml
  ```

in every epoch fake images and real images will be saved in `log/WGAN/{epoch}` in resolution of 64x64, you could check the training precess of WGAN at any time, and checkpoints will be saved in `checkpoints/WGAN`

**ATTENTION**: Actually this project is a bit complex than its introduction, if you want to train a model by yourself, see more information in [train-info.md](./train-info.md)

### Generate an anime picture

- use WGAN model as

  ```bash
  python generate.py --config-file configs/WGAN.yaml -g checkpoints/WGAN/{MODEL-WEIGHT-NAME}.pth
  ```

- use WGANP model as

  ```bash
  python generate.py --config-file configs/WGANP.yaml -g checkpoints/WGANP/{MODEL-WEIGHT-NAME}.pth
  ```

by default it will generate an 8x8 grid fake anime image under `./image`

Arguments:

- `-g`: short for generator, one argument with the name of the model weights
- `-s`: short for separate, no argument. Use `-s` to separate the grid image into 64 images

## Pretrained model

|model|dataset|Discriminator|Generator|
|:--:|:--:|:--:|:--:|
|WGAN|ANIME256|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_D_ANIME256.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_G_ANIME256.pth)|
||ANIME|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_D_ANIME.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGAN_G_ANIME.pth)|
|WGANP|ANIME256|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_D_ANIME256.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_G_ANIME256.pth)|
||ANIME|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_D_ANIME.pth)|[download](https://github.com/luzhixing12345/Anime-WGAN/releases/download/v0.0.4/WGANP_G_ANIME.pth)|

If you don't want to train by yourself(about 36~48h), download the pretrained model G and move it under `./checkpoints`, then run generate.py as above to generate images, remember to use the correct path and model name.

> actually if you just want to generate an anime picture you only need to download G, but whatever, I uploaded all.

## Detail information about this project

If you are familiar with **Chinese**, you could visit [my blog](https://luzhixing12345.github.io/tags/GAN/) which recorded my understanding of GAN, DCGAN, WGAN-CP, WGAN-GP, the evaluation method of GAN, and the whole project.

Sorry for my poor english, translate it into english if you really need it.

## Result

- generated images
- latent walk

  |modelWGAN + anime256|WGAN + anime|
  |:--:|:--:|
  |![1](https://raw.githubusercontent.com/learner-lu/picbed/master/walking_latent_space.gif)|![asdoqi](https://raw.githubusercontent.com/learner-lu/picbed/master/asdoqi.gif)|

  |WGANP + anime256|WGANP + anime|
  |:--:|:--:|
  |![asdhjono](https://raw.githubusercontent.com/learner-lu/picbed/master/asdhjono.gif)|![hcouga](https://raw.githubusercontent.com/learner-lu/picbed/master/hcouga.gif)|


## Conclusion

## Some extra work

if you have any other question, leave your confusion in Issue and I will apply as soon as possible.

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