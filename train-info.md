# Train-information



After each epoch, fake images and real images will be saved in `log/WGAN/{epoch}` in resolution of 64x64, you could check the training precess at any time, and checkpoints will be saved in `checkpoints/WGAN`

After training, it will generate some fake images in `./images`, and you could also get the `walking_latent_space.gif` of the finial model and `WGAN_process.gif` during the whole GAN training process.

All information will be saved






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

2. CUDA out of memory

   this happened when your GPU doesn't have enough memory to save the tensor

   try to use smaller batchsize