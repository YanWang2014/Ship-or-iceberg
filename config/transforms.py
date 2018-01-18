# https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
# https://github.com/ncullen93/torchsample/tree/master/torchsample/transforms
# https://keras.io/preprocessing/image/

'''
https://zhuanlan.zhihu.com/p/29513760

resize
rescale
noise
flip
rotate
shift
zoom
shear
contrast
channel shift
PCA
gamma
'''

import torch
from torchvision import transforms
import random
from PIL import Image
from .transforms_master import ColorJitter, scale, ten_crop, to_tensor, pad, RandomVerticalFlip
import collections
import torchsample
import cv2
import numpy as np

#input_size = 224 
#train_scale = 256 
#test_scale = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#normalize_ship = transforms.Normalize(mean=[-0.0810, -0.1032, -0.0921], std=[ 0.020395, 0.013316, 0.014961]) # rerange them to [-1, +1]
normalize_ship = transforms.Normalize(mean=[-0.0810, -0.1032, -0.0810], std=[ 0.020395, 0.013316, 0.020395]) # rerange them to [-1, +1]

def my_transform(img, input_size, train_scale, test_scale):
    img = scale(img, test_scale)
    imgs = ten_crop(img, input_size)  # this is a list of PIL Images
    return torch.stack([normalize(to_tensor(x)) for x in imgs], 0) # returns a 4D tensor
class my_ten_crops(object):
    def __init__(self, input_size, train_scale, test_scale):
        self.input_size = input_size
        self.train_scale = train_scale
        self.test_scale = test_scale
    def __call__(self, img):
        return my_transform(img, self.input_size, self.train_scale, self.test_scale)

# following ResNet paper, note that center crop should be removed if we can handle different image sizes in a batch
def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
class HorizontalFlip(object):
    def __init__(self, flip_flag):
        self.flip_flag = flip_flag
    def __call__(self, img):
        if self.flip_flag:
            return hflip(img)
        else:
            return img
def my_transform_multiscale_test(varied_scale, flip_flag):  
    return transforms.Compose([
        transforms.Scale(varied_scale),  
        transforms.CenterCrop(varied_scale), 
        HorizontalFlip(flip_flag),
        transforms.ToTensor(),
        normalize
    ])


def my_resize(img, size, interpolation=Image.BILINEAR):
    """ Adapted from but opposite to the oficial resize function, i.e.
        If size is an int, the larger edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size, size * width / height)
    """
    if isinstance(size, int):
        w, h = img.size
        if (w >= h and w == size) or (h >= w and h == size):
            return img
        if w > h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
class my_Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        return my_resize(img, self.size, self.interpolation)
    
class Pad2Set(object):
    """ Adapted from but different to the oficial Pad class, i.e.
    Pad the given PIL Image on all sides to the target value while keeping the original image
    in the center.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on w and h respectively. 
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    """

    def __init__(self, padding, fill=0):
        if isinstance(padding, collections.Sequence) and len(padding) not in [2]:
            raise ValueError("Padding must be an int or a 2 element tuple, not a " +
                             "{} element tuple".format(len(padding)))
        self.padding = padding
        self.fill = fill
    def __call__(self, img):
        #own code here, note that self.padding refers to the target size first, and
        #then goes back to its original meaning defined in the official pad class or function.
        #i.e. if a tuple of length 4 is provided
        #    this is the padding for the left, top, right and bottom borders respectively.
        w, h = img.size
        self.padding2 = ((self.padding-w)//2, (self.padding-h)//2, self.padding-w-(self.padding-w)//2, self.padding-h-(self.padding-h)//2)
        return pad(img, self.padding2, self.fill)

class numpy_Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return self.numpy_resize(img)
    
    def numpy_resize(self, img):
        return cv2.resize(img, (self.size, self.size))
    

class numpy_Shift_intensity(object):
    def __init__(self, mul = 0.2, plus_ratio = 0.2):
        self.mul = mul
        self.plus_ratio = plus_ratio
    def __call__(self, img):
        return self.numpy_shift_intensity(img)
    
    def numpy_shift_intensity(self, img):
        a = 1 + (np.random.random()*2 -1) * self.mul
        b = (np.random.random()*2-1) * self.plus_ratio
        img += 0.1*b
        img *= a
        return img
    
composed_data_transforms = {}
def data_transforms(phase, input_size = 224, train_scale = 256, test_scale = 256):
#    if phase == 'train2_ship':
#        print('Transforms are performed on numpy manually in json_dataset.py')
#    else:
    print('input_size %d, train_scale %d, test_scale %d' %(input_size,train_scale,test_scale))
    
    composed_data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize
    ]),
    'train2': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        ColorJitter(),
        transforms.ToTensor(), 
        normalize
    ]),
    'multi_scale_train': transforms.Compose([   ## following ResNet paper, but not include the standard color augmentation from AlexNet
        transforms.Scale(random.randint(384, 640)),  # May be adjusted to be bigger
        transforms.RandomCrop(input_size),  # not RandomSizedCrop
        transforms.RandomHorizontalFlip(), 
        ColorJitter(), # different from AlexNet's PCA method which is adopted in the ResNet paper?
        transforms.ToTensor(), 
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'ten_crop': my_ten_crops(input_size, train_scale, test_scale),#todo: merge my_transform
    'scale_pad': transforms.Compose([ 
        my_Resize(test_scale),   # 将长边scale到test_scale，保持长宽比
        Pad2Set(input_size), # pad 成正方形，边长为input_size
        transforms.ToTensor(),
        normalize
    ]),
    
    'train_ship': transforms.Compose([
#        numpy_Shift_intensity(mul = 0.5, plus_ratio = 0.5), 
#        numpy_Resize(input_size),
        transforms.ToTensor(), 
#        torchsample.transforms.RandomAffine(rotation_range=10, translation_range=[0.05, 0.05], shear_range=None, zoom_range=None),
        torchsample.transforms.RandomFlip(h=True, v=True, p=0.5)
#        normalize_ship
    ]),
    'val_ship': transforms.Compose([
#        numpy_Resize(input_size),
        transforms.ToTensor()
#        normalize_ship
    ])
    }
    return composed_data_transforms[phase]
