# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
from torchvision import transforms
import torch


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, f"log_{name}.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class logger:
    
    '''
    log in pytorch training
    '''
    def __init__(self,log_dir) -> None:
        
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(os.path.join(self.log_dir))
        
        self.d_loss = []
        self.g_loss = []
        self.iterations = []
        
    def log_losses(self,info, generator_iter):
        
        self.d_loss.append(info['d_loss'])
        self.g_loss.append(info['g_loss'])
        self.iterations.append(generator_iter)
        
    def log_images(self,info, generator_iter):
        
        generated_images = info['generated_images']
        real_images = info['real_images']

        if not os.path.exists(os.path.join(self.log_dir,'generated_images')):
            os.makedirs(os.path.join(self.log_dir, str(generator_iter)))
            
        # save images
        toPIL = transforms.ToPILImage()
        index = 0
        for generated_image, real_image in zip(generated_images, real_images):

            # inverse normalization
            # generated_image = generated_image * 0.5 + 0.5
            # from [-1,1] to [0,1]
            generated_image = torch.tensor(generated_image).mul(0.5).add(0.5)
            real_image = torch.tensor(real_image).mul(0.5).add(0.5)
        
            fake_image = toPIL(generated_image)
            real_image = toPIL(real_image)
            fake_image.save(os.path.join(self.log_dir, str(generator_iter), f'fake_image_{index}.png'))
            real_image.save(os.path.join(self.log_dir, str(generator_iter), f'real_image_{index}.png'))
            index += 1
            
        print('successfully save images')
        
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