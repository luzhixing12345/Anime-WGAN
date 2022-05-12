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
from utils.logger import logger
from utils.evaluate import get_inception_score
from torchvision import transforms,utils
from itertools import chain


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
        self.evaluate_iteration = cfg.SOLVER.EVALUATE_ITERATION
        self.evaluate_batch = cfg.SOLVER.EVALUATE_BATCH
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

        # change process
        self.fake_images = []
        self.save_number = cfg.IMAGE.SAVE_NUMBER
        self.prepare_number = cfg.IMAGE.PREPARE_NUMBER
        self.save_row_number = cfg.IMAGE.SAVE_ROW_NUMBER
        self.noise = torch.randn(self.save_number, self.G_input_size,1,1).to(self.device)
        
        # evaluation
        self.max_inception_score = 0
        self.best_model_path = ''
        
    def save_model(self,epoch):
        inception_score = self.evaluate_generator()
        if inception_score > self.max_inception_score:
            self.max_inception_score = inception_score
            self.best_model_path = os.path.join(self.model_checkpoint_dir, f'{self.cfg.PROJECT_NAME}_D_epoch_{epoch}.pth')
            print("New best model! Saving to {}".format(self.best_model_path))
        
        torch.save(self.G.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_G_epoch_{}.pth'.format(self.cfg.PROJECT_NAME,epoch)))
        torch.save(self.D.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_D_epoch_{}.pth'.format(self.cfg.PROJECT_NAME,epoch)))
        print('Models save to {}'.format(self.model_checkpoint_dir))

    def load_model(self):
        D_model_path = self.cfg.MODEL.D.PATH
        G_model_path = self.cfg.MODEL.G.PATH
        
        # if D_model_path != "":
        #     self.D.load_state_dict(torch.load(D_model_path))
        #     print("D_model loaded from {}".format(D_model_path))
        if G_model_path != "":
            self.G.load_state_dict(torch.load(G_model_path))
            print("G_model loaded from {}".format(G_model_path))
        if D_model_path != "":
            self.D.load_state_dict(torch.load(D_model_path))
            print("D_model loaded from {}".format(D_model_path))
    
    def generate_images(self):
        
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            z = torch.randn(self.prepare_number, self.G_input_size,1,1).to(self.device)
            z = z.to(self.device)
            fake_images = self.G(z)
            scores = self.D(fake_images).flatten()
            # use images which have high scores top self.save_number
            scores, indices = torch.sort(scores, descending=True)
            best_indices = indices[:self.save_number]
            worst_indices = indices[-self.save_number:]
            
            best_fake_images = fake_images[best_indices]
            worst_fake_images = fake_images[worst_indices]
            
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)   
            
        best_fake_images = best_fake_images.mul(0.5).add(0.5).squeeze().cpu()
        worst_fake_images = worst_fake_images.mul(0.5).add(0.5).squeeze().cpu()
        
        best_image_grid = utils.make_grid(best_fake_images, nrow=self.save_row_number)
        worst_image_grid = utils.make_grid(worst_fake_images, nrow=self.save_row_number)
        
        utils.save_image(best_image_grid, os.path.join(self.image_save_path, '{}_best.png'.format(self.cfg.PROJECT_NAME)))
        utils.save_image(worst_image_grid, os.path.join(self.image_save_path, '{}_worst.png'.format(self.cfg.PROJECT_NAME)))
        # save images one by one
        # toPIL = transforms.ToPILImage()
        # for i, image in enumerate(fake_images):
        #     image = image.mul(0.5).add(0.5)
        #     image = toPIL(image)
        #     image.save(os.path.join(self.image_save_path, '{}.png'.format(i)))

        
    def train(self):
        '''
        overload this method to train your model
        '''
        raise NotImplementedError
    
    def record_fake_images(self):
        
        with torch.no_grad():
            fake_images = self.G(self.noise)
            fake_images = fake_images.mul(0.5).add(0.5).cpu()
            image_grid = utils.make_grid(fake_images, nrow=self.save_row_number)
            self.fake_images.append(image_grid)
        
    def save_gif(self):
        gif_name = f'{self.cfg.MODEL.NAME}_epoch_{self.epochs}.gif'
        imageio.mimsave(os.path.join(self.log_dir, gif_name), self.fake_images)
        print('Gif saved to {}'.format(os.path.join(self.log_dir, gif_name)))

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
        return inception_score
    
    def save_best_model(self):
        
        # copy best model to current directory
        if self.best_model_path != '':
            print('best model saved to {}'.format(self.best_model_path) + ':' + str(self.max_inception_score))
            print('Copying best model to current directory')
            shutil.copy(self.best_model_path, os.path.join(self.model_checkpoint_dir, 'best_model.pth'))
            print('Copying best model to current directory done')
        else:
            print('No best model found')