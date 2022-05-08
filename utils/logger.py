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
        
    def log_losses(self,info, epoch):
        
        self.d_loss.append(info['d_loss'])
        self.g_loss.append(info['g_loss'])
        self.iterations.append(epoch)
        
    def log_images(self,info, epoch):
        
        real_images = info['real_images']
        fake_images = info['fake_images']
        real_labels = info['real_labels']
        fake_labels = info['fake_labels']

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
            
            if fake_labels[index]>0.5:
                fake_image.save(os.path.join(self.log_dir, epoch, f'fake_image_{index}_T.png'))
            else:
                fake_image.save(os.path.join(self.log_dir, epoch, f'fake_image_{index}_F.png'))
            if real_labels[index]>0.5:
                real_image.save(os.path.join(self.log_dir, epoch, f'real_image_{index}_T.png'))
            else:
                real_image.save(os.path.join(self.log_dir, epoch, f'real_image_{index}_F.png'))
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