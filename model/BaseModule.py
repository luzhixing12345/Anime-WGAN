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

class BasicModel(nn.Module):
    def __init__(self,cfg):
        super(BasicModel, self).__init__()
        self.cfg = cfg
        self.height = cfg.IMAGE.HEIGHT
        self.width = cfg.IMAGE.WIDTH
        self.channels = cfg.IMAGE.CHANNEL
        self.output_size = self.height * self.width * self.channels
        self.epochs = cfg.SOLVER.EPOCHS
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.device = cfg.MODEL.DEVICE
        self.model_checkpoint_dir = os.path.join(cfg.MODEL.CHECKPOINT_DIR, cfg.PROJECT_NAME)
        self.checkpoint_freq = cfg.SOLVER.CHECKPOINT_FREQ
        self.logger = logger(os.path.join(cfg.OUTPUT_DIR, cfg.PROJECT_NAME))

        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir)

    def reshape_images(self,data):
        
        number_of_images = self.cfg.IMAGE.NUMBER
        
        data = data.cpu().detach().numpy()[:number_of_images]
        images = []
        for sample in data:
            images.append(sample.reshape(self.channels, self.height, self.width))
        return images

    def save_model(self,epoch,iteration):
        torch.save(self.G.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_G_epoch_{}_iter_{}.pth'.format(self.cfg.PROJECT_NAME,epoch,iteration)))
        torch.save(self.D.state_dict(), os.path.join(self.model_checkpoint_dir, '{}_D_epoch_{}_iter_{}.pth'.format(self.cfg.PROJECT_NAME,epoch,iteration)))
        print('Models save to {}'.format(self.model_checkpoint_dir))

    def load_model(self, D_model_filename = "", G_model_filename = ""):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        
        if D_model_filename != "":
            self.D.load_state_dict(torch.load(D_model_path))
            print("D_model loaded from {}".format(D_model_path))
        if G_model_filename != "":
            self.G.load_state_dict(torch.load(G_model_path))
            print("G_model loaded from {}".format(G_model_path))
            
        
    def train(self):
        '''
        overload this method to train your model
        '''
        raise NotImplementedError