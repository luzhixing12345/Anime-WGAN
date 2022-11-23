'''
*Copyright (c) 2022 All rights reserved
*@description: train GAN model
*@author: Zhixing Lu
*@date: 2022-05-06
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

from config.config import get_cfg
from utils import *
from model.BaseModule import BasicGAN

def main():
    
    cfg = get_cfg()
    cfg = project_preprocess(cfg)

    train_dataloader = preprare_dataloader(cfg)
    
    model:BasicGAN = build_model(cfg)
    model.train(train_dataloader)
    model.generate_images()
    model.save_gif()

if __name__ == '__main__':
    main()
