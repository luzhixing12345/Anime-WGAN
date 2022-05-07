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
import os
from utils.logger import logger
from torchvision import transforms

class BasicModel(nn.Module):
    def __init__(self,cfg):
        super(BasicModel, self).__init__()
        self.cfg = cfg
        # input image
        self.height = cfg.IMAGE.HEIGHT
        self.width = cfg.IMAGE.WIDTH
        self.channels = cfg.IMAGE.CHANNEL
        self.input_size = self.channels * self.height * self.width
        # some hyperparameters
        self.epochs = cfg.SOLVER.EPOCHS
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        # some default parameters
        self.device = cfg.MODEL.DEVICE
        self.model_checkpoint_dir = os.path.join(cfg.MODEL.CHECKPOINT_DIR, cfg.PROJECT_NAME)
        self.checkpoint_freq = cfg.SOLVER.CHECKPOINT_FREQ
        self.logger = logger(os.path.join(cfg.OUTPUT_DIR, cfg.PROJECT_NAME))
        self.number_of_images = self.cfg.IMAGE.NUMBER
        self.image_save_path = cfg.IMAGE.SAVE_PATH
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

    def reshape_images(self,data):
        '''
        reshape images to (batch_size, channels, height, width)
        '''
        data = data.cpu().detach().numpy()[:self.number_of_images]
        images = []
        for sample in data:
            images.append(sample.reshape(self.channels, self.height, self.width))
        return images

    def save_model(self,epoch,iteration):
        torch.save(self.G.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_G_epoch_{}_iter_{}.pth'.format(self.cfg.PROJECT_NAME,epoch,iteration)))
        torch.save(self.D.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_D_epoch_{}_iter_{}.pth'.format(self.cfg.PROJECT_NAME,epoch,iteration)))
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