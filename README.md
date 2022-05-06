# GAN

## Inroduction

[Generative adversial nets](https://arxiv.org/abs/1406.2661) was published in 2014 NIPS

[DCGAN](https://arxiv.org/pdf/1511.06434.pdf) which using CNN in GAN has made a huge improvement.

## Relevant reference

- [知乎-GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)
- [GAN video by Li mu](https://www.bilibili.com/video/BV1rb4y187vD)
- [Github-GAN-tensorflow](https://github.com/YadiraF/GAN)
- [Github-GAN-pytorch](https://github.com/eriklindernoren/PyTorch-GAN)
- [KL散度](https://zhuanlan.zhihu.com/p/365400000)

## [Knowledge of GAN](GAN.md)

## Requirements

- python : 3
- pytorch
- other dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Dataset(Anime)

![20220506121357](https://raw.githubusercontent.com/learner-lu/picbed/master/20220506121357.png)

download the dataset zip and unzip it as `anime` in the the folder

- [Github-download](https://github.com/luzhixing12345/GAN/releases/download/v0.0.2/faces.zip)

  > try to use [Github proxy](https://ghproxy.com/) if too slow

## Use

### Step-1: Train

```bash
python train.py
```

### Step-2: Generate an anime picture

```bash
python generate.py
```

## Pretrained model

|D-epoch|download|G-epoch|download|
|:--:|:--:|:--:|:--:|
|D-100|[download]()|G-100|[download]()|
|D-200|[download]()|G-200|[download]()|
|D-300|[download]()|G-300|[download]()|
|D-400|[download]()|G-400|[download]()|
|D-500(best)|[download]()|G-500(best)|[download]()|

download the pretrained model D and G and move it under `./model`

> actually if you just want to generate an anime picture you only need to download G
>
> D could be used to discriminate whether the picture is real or not

- generate an anime picture

  ```bash
  python generate.py
  ```

- discriminate a picture

  ```bash
  python discriminate.py -i example-01-T.jpg
  ```

## Result

## Conclusion

## [Lab report]()

> actually this is my closing report of Data-science-introduction lesson, see more infomation in another [repository](https://github.com/luzhixing12345/data-science-introduction)
