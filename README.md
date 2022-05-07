# GAN

## Inroduction

[Generative adversial nets](https://arxiv.org/abs/1406.2661) was published in 2014 NIPSd

[DCGAN](https://arxiv.org/pdf/1511.06434.pdf) which using CNN in GAN has made a huge improvement.

TODO: an online website

## Requirements

- python : 3
- pytorch
- yacs `pip install yacs`

## Dataset(Anime)

![20220507024303](https://raw.githubusercontent.com/learner-lu/picbed/master/20220507024303.png)![20220507024427](https://raw.githubusercontent.com/learner-lu/picbed/master/20220507024427.png)

download the dataset zip and unzip it under `dataset` as `dataset/anime`

total dataset contains about 50000+ images of 96x96

- [Github-download](https://github.com/luzhixing12345/GAN/releases/download/v0.0.2/faces.zip)

  > try to use [Github proxy](https://ghproxy.com/) if too slow

## Use

### Train

```bash
python train.py
```

in every epoch fake images and real images will be saved in `log/DCGAN/{iteration}`, and checkpoints will be saved in `checkpoints/`

> see more information if you want to change default arguments in [train-info.md](./train-info.md)

### Generate an anime picture

```bash
python generate.py MODEL.G.PATH './checkpoints/DCGAN/xxx.pth'
```

by default it will generate 10 fake anime images under `./image`

if you want to change the number of image or change the image position

```bash
python generate.py MODEL.G.PATH './checkpoints/DCGAN/xxx.pth' IMAGE.NUMBER 20 IMAGE.SAVE_PATH 'another_path'
```

## Pretrained model

|D-epoch|download|G-epoch|download|
|:--:|:--:|:--:|:--:|
|D-300(best)|[download]()|G-300(best)|[download]()|

download the pretrained model D and G and move it under `./checkpoints`, then run generate.py as above to generate images

> actually if you just want to generate an anime picture you only need to download G

## Result

## Conclusion

## Some questions you may ask and some problems you may encounter

1. How to train with my own dataset?

   see [train-info.md](train-info.md)

2. The result doesn't seem well...

   Well, that's hard to explain, I'm not an expert of ML/DL, find other model? Or you may want to see the relevant reference below to find other excellent models

## Some errors you may encounter

1. BrokenPipeError: [Errno 32] Broken pipe

   a bug while multiple threads in Windows, add `DATALOADER.NUM_WORKERS 0`

   ```bash
   python train.py DATALOADER.NUM_WORKERS 0
   ```

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
