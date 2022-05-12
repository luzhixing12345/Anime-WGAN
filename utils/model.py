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

    total_model = {
        'GAN': GAN,
        'DCGAN': DCGAN,
        'WGAN64': WGAN64,
        'WGAN256': WGAN256,
        'WGAN64P': WGAN64P,
    }
    model = total_model[cfg.MODEL.NAME](cfg)

    if cfg.MODEL.DEVICE == 'cuda':
        model = model.cuda()
    
    return model