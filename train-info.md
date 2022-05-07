# Train-information

## About training configuration

I don't like to use parameters carried by long command. All the configuration infomation is in [config/defaults.py](config/defaults.py). I think the name of each parameter is clear enough for you to understand what it infers. You may change as you like, batch size, epochs and so on.

**BUT PAY ATTENTION THAT** if you want to change the size of image: `IMAGE.HEIGHT` or `IMAGE.WIDTH`. It's a little complex, you also need to change the network structure to fix your new size, I wrote some annotations in [DCGAN.py](model/DCGAN.py). There is the expected size of each layer in generator and discriminator. You need to calculate and change the network structure to fix your new size. Wish my annotations would help.

> That's why the origin dataset is 96x96 and I resize it as 64x64

In the paper of DCGAN, there are some tricks such as lr and betas of optimizer which are declared in [configs/DCGAN.yaml](configs/DCGAN.yaml)

## About GAN

You may notice that there is GAN in this project, yes, that's traditional GAN, but as we all know that traditional GAN is not easy to train, and the final result is also much worse than DCGAN. Actually this is my closing report of `Data-science-introduction lesson`, and I plan to train tranditional GAN first, however, you know the result was so terrible that I could not accept. So then I try DCGAN.

If you have interest in training GAN, you could use

```bash
python train.py --config-file ./configs/GAN.yaml
```

## About train with new dataset

Actually DCGAN could be used in many domains, using anime dataset to train, hh, that's just my hobby and I think generate an anime image is interesting.

Back to the subject

- first you need to prepare your dataset, and move it under `./dataset` such as `./dataset/house`
- change dataset name of [configs/DCGAN.yaml](configs/DCGAN.yaml)
- train as usual to get a 64x64 image

If you want to change the size of image or configuration, see `About training configuration` above.
