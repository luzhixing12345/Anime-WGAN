# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import logging.config
import os
from torchvision import transforms
import torch

def set_logger(cfg):
    logging.config.fileConfig(cfg.LOG_CONFIGURATION)
    logger = logging.getLogger('LOGGER')
    
    file_handler = logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, f"{cfg.PROJECT_NAME}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(file_handler)
    
    
def get_logger():
    return logging.getLogger('LOGGER')

class Logger:
    
    '''
    log in pytorch training
    '''
    def __init__(self,cfg) -> None:
        
        self.log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.PROJECT_NAME)
        if not os.path.exists(self.log_dir):
            os.makedirs(os.path.join(self.log_dir))
        
        self.logger = get_logger()
        self.d_loss = []  
        self.g_loss = []
        self.iterations = []
        
    def log_losses(self,info, epoch):
        
        self.d_loss.append(info['d_loss'])
        self.g_loss.append(info['g_loss'])
        self.iterations.append(epoch)
        
    def log_images(self,info, epoch):
        
        real_images = info['real_images']
        fake_images = info['fake_images']

        epoch = 'epoch_'+str(epoch)
        if not os.path.exists(os.path.join(self.log_dir,epoch)):
            os.makedirs(os.path.join(self.log_dir, epoch))
            
        # save images
        toPIL = transforms.ToPILImage()
        index = 0
        for fake_image, real_image in zip(fake_images, real_images):

            # inverse normalization
            # generated_image = generated_image * 0.5 + 0.5
            # from [-1,1] to [0,1]
            fake_image = torch.tensor(fake_image).mul(0.5).add(0.5)
            real_image = torch.tensor(real_image).mul(0.5).add(0.5)
        
            fake_image = toPIL(fake_image)
            real_image = toPIL(real_image)
            
            fake_image.save(os.path.join(self.log_dir, epoch, f'fake_image_{index}.png'))
            real_image.save(os.path.join(self.log_dir, epoch, f'real_image_{index}.png'))

            index += 1
            
        self.log('successfully save images')
        
    def save(self):
            
        with open(os.path.join(self.log_dir,'d_loss.txt'),'w') as f:
            for d_loss in self.d_loss:
                f.write(str(d_loss)+'\n')
        with open(os.path.join(self.log_dir,'g_loss.txt'),'w') as f:
            for g_loss in self.g_loss:
                f.write(str(g_loss)+'\n')
        with open(os.path.join(self.log_dir,'iterations.txt'),'w') as f:
            for iteration in self.iterations:
                f.write(str(iteration)+'\n')
        
        print('successfully save losses')
        
    def log(self,info):
        self.logger.info(info)