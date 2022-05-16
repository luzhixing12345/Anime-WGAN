# Train-information

I'm glad that somebody would like to follow this work. If it does help you, leave a :star: please.

## About training configuration

I don't like to use parameters carried by long command. All the configuration infomation is in [config/defaults.py](config/defaults.py). I think the name of each parameter is clear enough for you to understand what it infers. You may change as you like, batch size, epochs and so on.

logging part comes from python module `logging`, I think it's a more elegent way to log something instead of just using `print`.

Except the default configuration information, you **ONLY NEED TO FOUCES ON** detail configurations under folder `configs`, there are two `yaml` file `configs/WGAN.yaml` and `configs/WGANP.yaml`.

[WGAN.yaml](configs/WGAN.yaml) use some configurations different from [defaults.py](config/defaults.py), which needs to declared and it will overwrite the corresponding configurations while the program running.

So if you want to change the dataset from `anime256` to `anime`, you just need to change the yaml file.

If you don't want to steadily change file, you could also use the same yaml file but follow with the corresponding configuration like

```bash
# change dataset to anime
python train.py --config-file configs/WGAN.yaml DATASET.NAME anime

# change generator iterations to 50000
python train.py --config-file configs/WGANP.yaml MODEL.WGAN.GENERATOR_ITERS 50000
```

**BUT PAY ATTENTION THAT** if you want to change the size of image: `IMAGE.HEIGHT` or `IMAGE.WIDTH`. It's a little complex, you also need to change the network structure to fix your new size, I wrote some annotations in [WGAN.py](model/WGAN.py). There is the expected size of each layer in generator and discriminator. You need to calculate the expected input size and outpur size and change the network structure to fix your new size. Wish my annotations would help.

The whole process comes from my [python work flow](https://github.com/luzhixing12345/python-template)

---

You may notice that there is a DCGAN in the project, yes, but after training, DCGAN does not perform well in anime faces generating. So I abort this GAN model and try to use WGAN. In the paper of DCGAN, there are some tricks such as lr and betas of optimizer which are declared in [configs/DCGAN.yaml](configs/DCGAN.yaml). And WGAN also use the basic model structure of DCGAN.

## About train with new dataset

WGAN could be used in many domains, using anime dataset is just because I think generating an anime avatar is interesting.

Back to the subject

- first you need to prepare your dataset, and move it under `./dataset` such as `./dataset/house`
- change dataset name of [configs/WGAN.yaml](configs/WGAN.yaml) or use command line

If you want to change the size of image or configuration, see `About training configuration` above.

## About loss and IC score

Yes, that's a good question.
