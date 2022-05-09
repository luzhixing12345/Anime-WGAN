'''
*Copyright (c) 2022 All rights reserved
*@description: basic model
*@author: Zhixing Lu
*@date: 2022-05-07
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''


import torch
import torch.nn as nn
import imageio
import os
from utils.logger import logger
from torchvision import transforms,utils

class BasicGAN(nn.Module):
    def __init__(self,cfg):
        super(BasicGAN, self).__init__()
        self.cfg = cfg
        # input image
        self.height = cfg.IMAGE.HEIGHT
        self.width = cfg.IMAGE.WIDTH
        self.channels = cfg.IMAGE.CHANNEL
        # some hyperparameters
        self.epochs = cfg.SOLVER.EPOCHS
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        # discriminator
        self.D_dimension = cfg.MODEL.D.DIMENSION
        # generator
        self.G_dimension = cfg.MODEL.G.DIMENSION # 1024
        self.G_input_size = cfg.MODEL.G.INPUT_SIZE # 100
        # some default parameters
        self.device = cfg.MODEL.DEVICE
        self.model_checkpoint_dir = os.path.join(cfg.MODEL.CHECKPOINT_DIR, cfg.PROJECT_NAME)
        self.checkpoint_freq = cfg.SOLVER.CHECKPOINT_FREQ # default 1000
        self.log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.PROJECT_NAME)
        self.logger = logger(self.log_dir)
        self.number_of_images = self.cfg.IMAGE.NUMBER
        self.image_save_path = cfg.IMAGE.SAVE_PATH
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

        # change process
        self.fake_images = []
        self.gif_number = cfg.IMAGE.GIF_NUMBER
        self.gif_row_number = cfg.IMAGE.GIF_ROW_NUMBER
        self.noise = torch.rand(self.gif_number, self.G_input_size, 1, 1)

    def save_model(self,epoch):
        torch.save(self.G.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_G_epoch_{}.pth'.format(self.cfg.PROJECT_NAME,epoch)))
        torch.save(self.D.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_D_epoch_{}.pth'.format(self.cfg.PROJECT_NAME,epoch)))
        print('Models save to {}'.format(self.model_checkpoint_dir))

    def load_model(self):
        # D_model_path = self.cfg.MODEL.D.PATH
        G_model_path = self.cfg.MODEL.G.PATH
        
        # if D_model_path != "":
        #     self.D.load_state_dict(torch.load(D_model_path))
        #     print("D_model loaded from {}".format(D_model_path))
        if G_model_path != "":
            self.G.load_state_dict(torch.load(G_model_path))
            print("G_model loaded from {}".format(G_model_path))
    
    def generate_images(self):
        
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(self.number_of_images, 100, 1, 1)
            z = z.to(self.device)
            fake_images = self.G(z)
            fake_images = self.reshape_images(fake_images)
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)
            
        toPIL = transforms.ToPILImage()
        for i, image in enumerate(fake_images):
            image = image.mul(0.5).add(0.5)
            image = toPIL(image)
            image.save(os.path.join(self.image_save_path, '{}.png'.format(i)))

        
    def train(self):
        '''
        overload this method to train your model
        '''
        raise NotImplementedError
    
    def record_fake_images(self):
        
        with torch.no_grad():
            fake_images = self.G(self.noise)
            fake_images = fake_images.mul(0.5).add(0.5)
            fake_images = fake_images.detach().cpu().numpy()
            image_grid = utils.make_grid(fake_images, nrow=self.gif_row_number)
            self.fake_images.append(image_grid)
        
    def save_gif(self):
        gif_name = f'{self.cfg.MODEL.NAME}_epoch_{self.epochs}.gif'
        imageio.mimsave(os.path.join(self.log_dir, gif_name), self.fake_images)
        print('Gif saved to {}'.format(os.path.join(self.log_dir, gif_name)))
