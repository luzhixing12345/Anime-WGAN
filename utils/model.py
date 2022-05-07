'''
*Copyright (c) 2022 All rights reserved
*@description: build a GAN model
*@author: Zhixing Lu
*@date: 2022-05-06
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

from model import *

def build_model(cfg):

    if cfg.MODEL.NAME == 'GAN':
        model = GAN(cfg)
    elif cfg.MODEL.NAME == 'DCGAN':
        model = DCGAN(cfg)
    else:
        raise NotImplementedError

    if cfg.MODEL.DEVICE == 'cuda':
        model = model.cuda()
    
    return model