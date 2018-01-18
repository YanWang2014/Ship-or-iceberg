#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
注意BN在train和eval时表现不一样。
'''

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
import json
from model import load_model
from config import data_transforms
import pickle
import csv
from params import *
import numpy as np
import utils.json_dataset as jd

batch_size = BATCH_SIZE

if phases[0] == 'test_A':
    test_root['is_iceberg'] = 1
    
if phases[0] == 'train':
    test_root = TRAIN_ROOT 


checkpoint_filename = arch + '_' + pretrained
multi_checks = []
'''
在这里指定使用哪几个epoch的checkpoint进行平均
'''
for epoch_check in ['41', '46']:   # epoch的列表，如['10', '20', 'best']
    multi_checks.append('checkpoint/' + checkpoint_filename + '_' + str(epoch_check)+'.pth.tar')

#best_check = 'checkpoint/' + checkpoint_filename + '_best.pth.tar' 

model_conv = load_model(arch, pretrained, use_gpu=use_gpu, num_classes=num_classes,  AdaptiveAvgPool=AdaptiveAvgPool, SPP=SPP, num_levels=num_levels, pool_type=pool_type, bilinear=bilinear, stage=stage, SENet=SENet,se_stage=se_stage,se_layers=se_layers, threshold_before_avg = threshold_before_avg)

for param in model_conv.parameters():
    param.requires_grad = False #节省显存

if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
    model_conv.features = nn.DataParallel(model_conv.features)
    model_conv.cuda()
else:
    model_conv = nn.DataParallel(model_conv).cuda()
    
    
def write_to_csv(aug_softmax, epoch_i = None): #aug_softmax[img_name_raw[item]] = temp[item,:]

    if epoch_i != None:
        file = 'result/'+ phases[0] +'_1_'+ epoch_i.split('.')[0].split('_')[-1] + '.csv'
    else:
        file = 'result/'+ phases[0] +'_1.csv'
    with open(file, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile,dialect='excel')
        writer.writerow(["id", "is_iceberg"])
        for item in aug_softmax.keys():
            writer.writerow([item, max(min(aug_softmax[item][1], 0.9999), 0.0001)])

transformed_dataset_test = jd.ImageDataset(test_root, include_target = True, X_transform = data_transforms(val_transform,input_size, train_scale, test_scale))          

dataloader = {phases[0]:DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS)
             }
dataset_sizes = {phases[0]: len(test_root)}


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
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: logits
    target: labels
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        

    pred_list = pred.tolist()  #[[14, 13], [72, 15], [74, 11]]
    return res, pred_list


def test_model (model, criterion):
    since = time.time()

    mystep = 0    

    for phase in phases:
        
        model.eval()  # Set model to evaluate mode

        top1 = AverageMeter()
        top3 = AverageMeter()
        loss1 = AverageMeter()
        aug_softmax = {}

        # Iterate over data.
        for dict_ in dataloader[phase]:
            # get the inputs
            mystep = mystep + 1
#            if(mystep%10 ==0):
#                duration = time.time() - since
#                print('step %d vs %d in %.0f s' % (mystep, total_steps, duration))

            inputs =  dict_['img']
            labels = dict_['target']
            img_name_raw = dict_['id']
            angle = dict_['angle']
            the_size = dict_['size']

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                size_var = Variable(the_size.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            if arch.startswith('modified'):
                outputs = model(inputs.float(), size_var.float())
            else:
                outputs = model(inputs)
            crop_softmax = nn.functional.softmax(outputs)
            temp = crop_softmax.cpu().data.numpy()
            for item in range(len(img_name_raw)):
                aug_softmax[img_name_raw[item]] = temp[item,:] #防止多线程啥的改变了图片顺序，还是按照id保存比较保险
                
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            
#            # statistics
            res, pred_list = accuracy(outputs.data, labels.data, topk=(1, 2))
            prec1 = res[0]
            prec3 = res[1]
            top1.update(prec1[0], inputs.size(0))
            top3.update(prec3[0], inputs.size(0))
            loss1.update(loss.data[0], inputs.size(0))
            

        print(' * Prec@1 {top1.avg:.6f} Prec@3 {top3.avg:.6f} Loss@1 {loss1.avg:.6f}'.format(top1=top1, top3=top3, loss1=loss1))

    return aug_softmax



criterion = nn.CrossEntropyLoss()


######################################################################
# val and test
total_steps = 1.0  * len(test_root) / batch_size * len(multi_checks)
print(total_steps)

class Average_Softmax(object):
    """for item in range(len(img_name_raw)):
        aug_softmax[img_name_raw[item]] = temp[item,:]
    """
    def __init__(self, inits):
        self.reset(inits)
    def reset(self, inits):
        self.val = inits
        self.avg = inits
        self.sum = inits
        self.total_weight = 0
    def update(self, val, w=1):
        self.val = val
        self.sum_dict(w)
        self.total_weight += w
        self.average()
    def sum_dict(self, w):
        for item in self.val.keys():
            self.sum[item] += (self.val[item] * w) 
    def average(self):
        for item in self.avg.keys():
            self.avg[item] = self.sum[item]/self.total_weight

image_names = test_root['id'].tolist()
inits = {}
for name in image_names:
    inits[name] = np.zeros(2)
aug_softmax_multi = Average_Softmax(inits)


for i in multi_checks:
    i_checkpoint = torch.load(i)
    print(i)
    if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
        #model_conv.features = nn.DataParallel(model_conv.features)
        #model_conv.cuda()
        model_conv.load_state_dict(i_checkpoint['state_dict']) 
    else:
        #model_conv = nn.DataParallel(model_conv).cuda()
        model_conv.load_state_dict(i_checkpoint['state_dict']) 
    aug_softmax = test_model(model_conv, criterion)
    write_to_csv(aug_softmax, i)
    aug_softmax_multi.update(aug_softmax)

'''
输出融合的结果，并计算融合后的loss和accuracy
'''
def cal_loss(aug_softmax, test_root):
    loss1 = 0
    for row in range(len(test_root)):
        loss1 -= np.log(aug_softmax[test_root['id'][row]][test_root['is_iceberg'][row]])
    loss1 /= len(test_root)
    print('Loss@1 {loss1:.6f}'.format(loss1=loss1))
write_to_csv(aug_softmax_multi.avg)
cal_loss(aug_softmax_multi.avg, test_root)  