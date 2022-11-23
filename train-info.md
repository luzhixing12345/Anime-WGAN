# Train-information

## 关于训练配置

我不喜欢使用长命令携带的参数.所有配置信息都在 [config/defaults.py](config/defaults.py)中

参数名相对明确,我相信您可以根据参数名称来推断其表示的含义,您可以根据需要更改batch size | learning rate等

logging 部分来自 python 模块 `logging`,这是一种更优雅的记录方式,如果您希望了解logging模块可以参考[优雅的日志记录-logging](https://luzhixing12345.github.io/2022/05/13/python/%E4%BC%98%E9%9B%85%E7%9A%84%E6%97%A5%E5%BF%97%E8%AE%B0%E5%BD%95-logging/)

除了默认的配置信息,**只需要关注**在文件夹`configs`下的详细配置,有两个`yaml`文件`configs/WGAN.yaml`和`config/WGAN.yaml`。

[WGAN.yaml](configs/WGAN.yaml)使用了一些不同于[defaults.py](config/defaults.py)的配置,程序运行时会覆盖相应的配置。

如果你想将切换数据集从 anime256 更改为 anime，你只需要更改 yaml 文件即可,我也推荐在yaml文件中修改而不是直接修改 default.py

例如从

```yaml
PROJECT_NAME: WGAN

DATASET:
  NAME: dataset/anime256
  TRAIN_TEST_RATIO: 1.0
```

到

```yaml
PROJECT_NAME: WGAN

DATASET:
  NAME: dataset/anime
  TRAIN_TEST_RATIO: 1.0
```

如果你不想不断地改变文件，你也可以使用相同的 yaml 文件, 然后在后面补充添加修改,比如

```bash
# change dataset to anime
python train.py --config-file configs/WGAN.yaml DATASET.NAME anime

# change generator iterations to 50000
python train.py --config-file configs/WGANP.yaml MODEL.WGAN.GENERATOR_ITERS 50000
```

- **注意**: 如果您想更改图像的大小：`IMAGE.HEIGHT` 或 `IMAGE.WIDTH`. 这个有点复杂，你还需要改变网络结构来搭配图像的新尺寸,我在[WGAN.py](model/WGAN.py)中写了一些注释。 生成器和鉴别器中的每一层都有预期的大小。 您需要计算预期的输入大小和输出大小并更改网络结构以修复您的新大小。 希望我的注释会有所帮助。

- **注意**: 如果你想使用同一个yaml文件训练两次或更多次，**记得使用唯一的PROJECT_NAME**，同一个PROJECT_NAME会覆盖之前的。

  ```bash
  python train.py --config-file configs/WGAN.yaml PROJECT_NAME MYTRAIN_1
  ```

  or change in yaml file

  ```yaml
  PROJECT_NAME: MYTRAIN_1

  DATASET:
    NAME: dataset/anime256
    TRAIN_TEST_RATIO: 1.0
  ```

整个项目的文件构成来自[python template](https://github.com/luzhixing12345/python-template)

> 你可能注意到项目中有一个 DCGAN.是的，但是经过训练，DCGAN 在动漫人脸生成方面表现不佳。 所以我中止了这个 GAN 模型并尝试使用 WGAN。 在DCGAN的论文中，在[configs/DCGAN.yaml](configs/DCGAN.yaml)中声明了优化器的lr和betas等tricks. 而WGAN也沿用了DCGAN的基本模型结构。

## About the visualization result

训练之际,整个日志信息都在`log/{PROJECT_NAME}.log`中，在每个checkpoint之后，你可以在`log/{PROJECT_NAME}/{iteration}`中看到假图和真图来帮助你检查GAN的训练过程。

您将在 `log/{PROJECT_NAME}` 中获得 `walking_latent_space.gif` 和 `{PROJECT_NAME}_process.gif`

你会在`log/{PROJECT_NAME}`中得到D和G的loss曲线

你会得到一张由`./images`中最后的G生成的网格图,更多生成图片的方式请参阅README。

## About train with new dataset

WGAN 可以用于许多领域，使用动漫数据集只是因为我认为生成动漫头像很有趣

如果您想使用自己的数据集

- 首先你需要准备你的数据集，并将它移动到 `./dataset` 下，例如 `./dataset/house`
- 更改 yaml 文件中的数据集名称或使用命令行更改(前文提及)
- 不要忘记使用独特的 PROJECT_NAME

如果您想更改图像或配置的大小，请参阅上面的“关于训练配置”。

## About loss and evluation index

是的，这是个好问题。

由于模型结构和损失函数不同，很难使用损失值来评估模型。有两个模型要训练，也许你的判别器训练得太好以至于生成的图像全都判断正确(这种情况占大多数),导致损失增加了. 或者你的判别器训练得太差了，同样的情况发生在生成器上。

GAN 模型还有其他评估分数。比如Inception score，FID等等。其实我已经用inception score来计算模型了,如果您想计算这个分数

- 首先，从 https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth 下载 Google 预训练的初始模型，并将其设置在 `checkpoints` 下
- 设置 MODEL.WGAN.IC = True

但是为什么我默认将此配置设置为 False 呢？那是因为IC分数在这个项目中看起来真的很奇怪。从4开始，到3.7、3.6、3.5，越来越低到3.4。

它应该在训练过程中增加，但为什么呢？我不明白。也许是因为 IC 分数的弱点？我不知道
