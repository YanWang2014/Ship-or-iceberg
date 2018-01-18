#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 27 14:19:10 2017

@author: wayne

"""
import torch
import pandas as pd
import numpy as np


def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])
def size(arr):     
    return float(np.sum(arr<-5))/(75*75)

'''
all data
'''
TRAIN_ROOT = pd.read_json('data/train.json')
TRAIN_ROOT = TRAIN_ROOT.iloc[np.random.permutation(len(TRAIN_ROOT))]
TRAIN_ROOT['inc_angle'] = pd.to_numeric(TRAIN_ROOT['inc_angle'], errors='coerce')
TRAIN_ROOT['iso1'] = TRAIN_ROOT.iloc[:, 0].apply(iso)
TRAIN_ROOT['iso2'] = TRAIN_ROOT.iloc[:, 1].apply(iso)
TRAIN_ROOT['s1'] = TRAIN_ROOT.iloc[:,5].apply(size)
TRAIN_ROOT['s2'] = TRAIN_ROOT.iloc[:,6].apply(size)
TRAIN_ROOT['band_1'] = TRAIN_ROOT['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
TRAIN_ROOT['band_2'] = TRAIN_ROOT['band_2'].apply(lambda x: np.array(x).reshape(75, 75))


'''
CV
'''
VALIDATION_ROOT = None


'''
测试用的数据
'''
#phases = ['test_A']
#if phases[0] == 'test_A':
#    test_root = pd.read_json('data/test.json')
#    test_root['inc_angle'] = pd.to_numeric(test_root['inc_angle'], errors='coerce')
#    test_root['iso1'] = test_root.iloc[:, 0].apply(iso)
#    test_root['iso2'] = test_root.iloc[:, 1].apply(iso)
#    test_root['s1'] = test_root.iloc['iso1'].apply(size)
#    test_root['s2'] = test_root.iloc['iso2'].apply(size)
#    test_root['band_1'] = test_root['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
#    test_root['band_2'] = test_root['band_2'].apply(lambda x: np.array(x).reshape(75, 75))


arch = 'modified_resnet18' # preact_resnet50, resnet152
pretrained = 'imagenet' #imagenet
num_classes = 2
threshold_before_avg =False
evaluate = False
checkpoint_filename = arch + '_' + pretrained
save_freq = 2
try_resume = False
print_freq = 10
if_debug = False
start_epoch = 0
class_aware = False
AdaptiveAvgPool = True
SPP = False
num_levels = 1 # 1 = fcn
pool_type = 'avg_pool'
bilinear = {'use':False,'dim':16384}  #没有放进hyper_board
stage = 2 #注意，1只训练新加的fc层，改为2后要用try_resume = True
SENet = False
se_layers = [None,None,None,'7'] # 4,5,6,7 [3, 8, 36, 3]
print(se_layers)
se_stage = 1 # 1冻结前面几层[去模型那调]， 2全开放
input_size = 75#[224, 256, 384, 480, 640] 
train_scale = 75
test_scale = 75
train_transform = 'train_ship'
val_transform = 'val_ship'

# training parameters:
BATCH_SIZE = 64
INPUT_WORKERS = 4
epochs = 60
use_epoch_decay = True # 可以加每次调lr时load回来最好的checkpoint
decay_epochs = [20, 40, 50]
lr = 1e-4  #0.01  0.001
lr_min = 1e-1
lr_decay = 0.5

if_fc = False #是否先训练最后新加的层，目前的实现不对。
lr1 = lr_min #if_fc = True, 里面的层先不动
lr2 = 0.2 #if_fc = True, 先学好最后一层
lr2_min = 0.019#0.0019 #lr2每次除以10降到lr2_min，然后lr2 = lr, lr1 = lr2/slow
slow = 1 #if_fc = True, lr1比lr2慢的倍数
print('lr=%.8f, lr1=%.8f, lr2=%.8f, lr2_min=%.8f'% (lr,lr1,lr2,lr2_min))

weight_decay = 0.005 #.05 #0.0005 #0.0001  0.05太大。试下0.01?
optim_type = 'Adam' #Adam SGD http://ruder.io/optimizing-gradient-descent/
confusions = None#'Entropic' #'Pairwise' 'Entropic'
confusion_weight = 0#0.1# 0.5  #for pairwise loss is 0.1N to 0.2N (where N is the number of classes), and for entropic is 0.1-0.5. https://github.com/abhimanyudubey/confusion
betas=(0.9, 0.999)
eps=1e-08 # 0.1的话一开始都是prec3 4.几
momentum = 0.9

use_gpu = torch.cuda.is_available()

triplet = False
