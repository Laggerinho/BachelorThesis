from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import utils
import PIL
import pickle
from tqdm import tqdm
import time

from . import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ClassificationModel_two_categories(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        # for 0 and 180 degrees if i%2 == 0
        # for 90 and 270 degrees if (i+1)%2 == 0
        # for 0 and 90 degrees if (i%4 == 0 or (i+3)%4 == 0)
        # for 90 and 180 degrees if ((i+3)%4 == 0 or (i+2)%4 == 0)
        self.inx = np.array([i for i in range(labels.size()[0]) if ((i+1)%4 == 0 or (i+2)%4 == 0)])
        self.inx = torch.from_numpy(self.inx)
        self.inx = self.inx.cuda()
        dataX = dataX[self.inx]
        # for 0 and 180 degrees divide by 2
        # for  90 and 270 substract 1 and divide by 2 (don't sub. by 1/2)
        # for 0 and 90 just the labels
        # for 90 and 180 substract 1
        labels = (labels[self.inx] - 2)
        # print(labels)
        batch_load_time = time.time() - start
        #********************************************************

        #********************************************************
        start = time.time()
        if do_train: # zero the gradients
            self.optimizers['model'].zero_grad()
        #********************************************************

        #***************** SET TORCH VARIABLES ******************
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train))
        # print(dataX_var.size())
        labels_var = torch.autograd.Variable(labels, requires_grad=False)
        # print(labels_var.size())
        #********************************************************

        #************ FORWARD THROUGH NET ***********************
        pred_var = self.networks['model'](dataX_var)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        record = {}
        loss_total = self.criterions['loss'](pred_var, labels_var)
        record['prec1'] = accuracy(pred_var.data, labels, topk=(1,))[0][0]
        record['loss'] = loss_total.data[0]
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
