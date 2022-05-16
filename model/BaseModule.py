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
import shutil
from utils.logger import Logger
from utils.evaluate import get_inception_score
from torchvision import transforms,utils
from itertools import chain
import numpy as np


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
        self.checkpoint_freq = cfg.SOLVER.CHECKPOINT_FREQ
        self.log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.PROJECT_NAME)
        self.logger = Logger(cfg)
        self.number_of_images = self.cfg.IMAGE.NUMBER
        self.image_save_path = cfg.IMAGE.SAVE_PATH
        self.image_separate = cfg.IMAGE.SEPARATE # default False, use -s to separate
        self.evaluate_iteration = cfg.SOLVER.EVALUATE_ITERATION
        self.evaluate_batch = cfg.SOLVER.EVALUATE_BATCH
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

        # change process
        self.GAN_process = [] # record in each checkpoint for self.G(self.noise)
        self.save_number = cfg.IMAGE.SAVE_NUMBER
        self.save_row_number = cfg.IMAGE.SAVE_ROW_NUMBER
        self.noise = torch.randn(self.save_number, self.G_input_size,1,1).to(self.device)
        
        # evaluation
        self.max_inception_score = 0
        
    def save_model(self,epoch):
        inception_score = self.evaluate_generator()
        if inception_score > self.max_inception_score:
            self.max_inception_score = inception_score
            # self.best_model_path = os.path.join(self.model_checkpoint_dir, f'{self.cfg.PROJECT_NAME}_D_epoch_{epoch}.pth')
            # self.logger.log("New best model! Saving to {}".format(self.best_model_path))
        
        torch.save(self.G.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_G_epoch_{}.pth'.format(self.cfg.PROJECT_NAME,epoch)))
        torch.save(self.D.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_D_epoch_{}.pth'.format(self.cfg.PROJECT_NAME,epoch)))
        self.logger.log('Models save to {}'.format(self.model_checkpoint_dir))

    def load_model(self):
        G_model_path = self.cfg.MODEL.G.PATH
        
        # if D_model_path != "":
        #     self.D.load_state_dict(torch.load(D_model_path))
        #     print("D_model loaded from {}".format(D_model_path))
        if G_model_path != "":
            self.G.load_state_dict(torch.load(G_model_path))
            self.logger.log("G_model loaded from {}".format(G_model_path))

    
    def generate_images(self):
        
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(self.save_number, self.G_input_size,1,1).to(self.device)
            z = z.to(self.device)
            fake_images = self.G(z)

            
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)   
            
        fake_images = fake_images.mul(0.5).add(0.5).cpu()
        
        if self.image_separate:
            # save images one by one
            toPIL = transforms.ToPILImage()
            for i, image in enumerate(fake_images):
                image = toPIL(image)
                image.save(os.path.join(self.image_save_path, '{}.png'.format(i)))
        else:
            image_grid = utils.make_grid(fake_images, nrow=self.save_row_number)
            utils.save_image(image_grid, os.path.join(self.image_save_path, '{}.png'.format(self.cfg.PROJECT_NAME)))
        

        
    def train(self):
        '''
        overload this method to train your model
        '''
        raise NotImplementedError
    
    def record_fake_images(self):
        
        with torch.no_grad():
            fake_images = self.G(self.noise)
            fake_images = fake_images.mul(0.5).add(0.5).cpu().data
            image_grid = utils.make_grid(fake_images, nrow=self.save_row_number)
            image_grid = np.transpose(image_grid.numpy(), (1, 2, 0))
            self.GAN_process.append(image_grid)
        
    def save_gif(self):
        gif_name = f'{self.cfg.PROJECT_NAME}_process.gif'
        imageio.mimsave(os.path.join(self.log_dir, gif_name), self.GAN_process)
        self.logger.log('Gif saved to {}'.format(os.path.join(self.log_dir, gif_name)))
        
        # only walking in the latent space for the finial model weight
        self.walking_latent_space()

    def evaluate_generator(self):
        sample_list = []
        for i in range(self.evaluate_iteration):
            z = torch.randn(self.evaluate_batch, self.G_input_size,1,1).to(self.device)
            with torch.no_grad():
                samples = self.G(z)
            sample_list.append(samples)
        new_sample_list = list(chain.from_iterable(sample_list))
        inception_score = get_inception_score(new_sample_list, 
                                              cuda=self.device == 'cuda',
                                              batch_size=self.batch_size,
                                              resize=True,
                                              splits = 10)
        self.logger.log('Inception score: {}'.format(inception_score))
        return inception_score
    
    # ABORT
    def save_best_model(self):
        
        # copy best model to current directory
        if self.best_model_path != '':
            self.logger.log('best model saved to {}'.format(self.best_model_path) + ':' + str(self.max_inception_score))
            self.logger.log('Copying best model to current directory')
            shutil.copy(self.best_model_path, os.path.join(self.model_checkpoint_dir, 'best_model.pth'))
            self.logger.log('Copying best model to current directory done')
        else:
            self.logger.log('No best model found')
        
        
    def walking_latent_space(self):
        '''
        walk through latent space using linear interpolation
        '''
        self.walk_step = self.cfg.WALKING_LATENT_SPACE.STEP
        self.walk_number = self.cfg.WALKING_LATENT_SPACE.IMAGE_NUMBER
        self.walk_row_number = self.cfg.WALKING_LATENT_SPACE.IMAGE_ROW_NUMBER
        self.walking_fps = self.cfg.WALKING_LATENT_SPACE.IMAGE_FPS
        
        walking_space_images = []
        
        # interpolate between twe noise(z1, z2).
        z1 = torch.randn(self.walk_number, self.G_input_size, 1, 1).to(self.device)
        z2 = torch.randn(self.walk_number, self.G_input_size, 1, 1).to(self.device)
        
        alpha_increase = 1.0 / self.walk_step
        alpha = 0
        for _ in range(self.walk_step+1):
            latent_vector = (1 - alpha) * z1 + alpha * z2
            with torch.no_grad():
                fake_im = self.G(latent_vector)
                fake_im = fake_im.mul(0.5).add(0.5).cpu().data #denormalize
            alpha = alpha + alpha_increase
            image_grid = utils.make_grid(fake_im, nrow=self.walk_row_number)
            img_grid = np.transpose(image_grid.numpy(), (1, 2, 0))
            walking_space_images.append(img_grid)

        imageio.mimsave(os.path.join(self.image_save_path, 'walking_latent_space.gif'), walking_space_images)
        self.logger.log('Walking latent space done, save to {}'.format(os.path.join(self.image_save_path, 'walking_latent_space.gif')))
        
        


        
        