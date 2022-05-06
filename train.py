'''
*Copyright (c) 2022 All rights reserved
*@description: train GAN model
*@author: Zhixing Lu
*@date: 2022-05-06
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

from config.config import get_cfg
from utils.basic_utils import project_preprocess
import logging



def main():
    
    cfg = get_cfg()
    cfg = project_preprocess(cfg)

    
    return


if __name__ == '__main__':
    main()
