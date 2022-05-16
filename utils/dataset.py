'''
*Copyright (c) 2022 All rights reserved
*@description: load dataset
*@author: Zhixing Lu
*@date: 2022-05-06
*@email: luzhixing12345@163.com
*@Github: luzhixing12345
'''

import os
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image

def preprare_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize(size=(cfg.IMAGE.HEIGHT, cfg.IMAGE.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGE.PIXEL_MEAN , std=cfg.IMAGE.PIXEL_STD)
    ])
    
    # load pytorch dataset
    if cfg.DATASET.NAME == 'MNIST':
        # overload the __getitem__ method: only return the image
        datasets.MNIST.__getitem__ = lambda self, index: self.transform(Image.fromarray(self.data[index].numpy(), mode='L'))
        dataset = datasets.MNIST(root="dataset",transform=transforms.Compose([
                                                    transforms.Resize(size=(cfg.IMAGE.HEIGHT, cfg.IMAGE.WIDTH)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=0.5 , std=0.5)
                                                    ]))
    else:
        dataset = GAN_dataset(root=cfg.DATASET.NAME, transform=transform)
    
    # split dataset into train and test
    # unsupervise learning only use train dataset, supervised learning use train and test dataset
    # by default DATASET.TRAIN_TEST_RATIO = 1
    if cfg.DATASET.TRAIN_TEST_RATIO!=1:
        train_size = int(len(dataset) * cfg.DATASET.TRAIN_TEST_RATIO)
        test_size = len(dataset) - train_size
        dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
    train_dataloader = DataLoader(dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
    return train_dataloader

class GAN_dataset(Dataset):
    
    def __init__(self,root,transform) -> None:
        super().__init__()
        
        self.root = root
        self.transform = transform

        self.file_name = []
        self.__load_data()
        
    def __load_data(self):
        '''
        unsupervised learning without label
        '''
        for root, dirs, files in os.walk(self.root):
            for file in files:
                self.file_name.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, index):
        file_name = self.file_name[index]
        image = Image.open(file_name)
        if self.transform:
            image = self.transform(image)
        return image