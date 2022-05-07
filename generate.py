'''
*Copyright (c) 2022 All rights reserved
*@description: generate images from GAN model
*@author: Zhixing Lu
*@date: 2022-05-08
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

from config.config import get_cfg
from utils import *

def main():
    
    cfg = get_cfg()
    cfg = project_preprocess(cfg)
    
    model = build_model(cfg)
    model.load_model()
    model.generate_images()
    print('Generate images successfully!')
    

if __name__ == '__main__':
    main()