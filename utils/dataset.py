'''
*Copyright (c) 2022 All rights reserved
*@description: load dataset
*@author: Zhixing Lu
*@date: 2022-05-06
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def preprare_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize(size=(cfg.IMAGE.HEIGHT, cfg.IMAGE.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=cfg.DATASET_NAME, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
    return dataloader