#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
adapted from https://www.kaggle.com/heyt0ny/pytorch-custom-dataload-with-augmentaion

https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/13
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rerange them to [-1, +1]

-0.0810
-0.1032

1.00000e-02 *
  2.0395
  1.3316

https://www.kaggle.com/supersp1234/tools-for-pytorch-transform

一个问题是只有两个channel，一个问题是数值为float的numpy而pytorch许多变换只作用于pil image
'''
import sys
sys.path.append('../')

import numpy as np 
import pandas as pd
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
import cv2
from torchvision import transforms
from PIL import Image
from config import transforms_master

class ImageDataset(data.Dataset):
    def __init__(self, X_data, include_target, X_transform = transforms.ToTensor()):

        self.X_data = X_data
        self.include_target = include_target
        self.X_transform = X_transform

    def __getitem__(self, index):
#        np.random.seed() 
        #get 2 channels of our image
        img1 = self.X_data.iloc[index]['band_1']
        img2 = self.X_data.iloc[index]['band_2']
        img3 = (img1+img2)/2.0

        img = np.stack([img1, img2, img3], axis = 2)
        img = img.astype(np.float32)


        #get angle and img_name
        angle = self.X_data.iloc[index]['inc_angle']
        img_id = self.X_data.iloc[index]['id']
        
        #perform augmentation
#        img = self.random_horizaontal_flip(img)
#        img = self.resize(img)
        if self.X_transform:
            img = self.X_transform(img)#, **{'u' : self.u})
            
        #so our loader will yield dictionary wi such fields:
        dict_ = {'img' : img,
                'id' : img_id, 
                'angle' : angle,
                }
        
        #if train - then also include target
        if self.include_target:
            target = self.X_data.iloc[index]['is_iceberg']
            dict_['target'] = target
        else:
            dict_['target'] = 1
            
        dict_['size'] = (self.X_data.iloc[index]['s1'] + self.X_data.iloc[index]['s2'])/2

        return dict_

    def __len__(self):
        return len(self.X_data)
        
    #custom aug function for numpy image:
    # horizontal flips, shifts and scale. https://www.jianshu.com/p/b5c29aeaedc7
    #https://github.com/ncullen93/torchsample/tree/master/torchsample/transforms
    #http://augmentor.readthedocs.io/en/master/
#    @staticmethod
#    def random_vertical_flip(img, u = 0.5):
#        if np.random.random() < u:
#            img = cv2.flip(img, 0)
#        return img
#    @staticmethod
#    def random_horizaontal_flip(img, u = 0.5):
#        if np.random.random() < u:
#            img = cv2.flip(img, 1)
#        return img
#    @staticmethod
#    def resize(img, size = 224):
#        return cv2.resize(img, (size, size))
#    @staticmethod
#    def rotate(img, deg = ):
#        return 
#    @staticmethod
#    def shift(img, dist = ):
#        return
    

if __name__ == "__main__":
    
    train = pd.read_json('../data/train.json')
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
    train['band_1'] = train['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    train['band_2'] = train['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
            
    batch_size = 10
    train_ds = ImageDataset(train, include_target = True, X_transform = transforms.ToTensor())
    USE_CUDA = False #for kernel
    THREADS = 4 #for kernel
    train_loader = data.DataLoader(train_ds, batch_size,
                                        shuffle=False,
                                        num_workers = THREADS,
                                        pin_memory= USE_CUDA )
    
    #calculate mean and variance
    class AverageMeter(object):
        def __init__(self):
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            
    mean_meter = AverageMeter()
    for i, dict_ in enumerate(train_loader):  # nchw
        image = dict_['img']
        if i%10 ==0:
            print(i)
        mean_meter.update(image.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
    
    mean = mean_meter.avg
    print(mean.squeeze())
    std_meter =  AverageMeter()
    for i, dict_ in enumerate(train_loader):  # nchw
        image = dict_['img']
        if i%10 ==0:
            print(i)
        std_meter.update(((image-mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), image.size(0))  
    print(std_meter.avg.squeeze().sqrt())