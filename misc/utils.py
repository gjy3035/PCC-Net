from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import shutil
from config import cfg
import pdb
import math



def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                n = module.weight.size(1)
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()


# def initialize_weights(models):
#     for model in models:
#         real_init_weights(model)


# def real_init_weights(m):

#     if isinstance(m, list):
#         for mini_m in m:
#             real_init_weights(mini_m)
#     else:
#         if isinstance(m, nn.Conv2d):    
#             nn.init.normal_(m.weight, std=0.01)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0.0, std=0.01)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         elif isinstance(m,nn.Module):
#             for mini_m in m.children():
#                 real_init_weights(mini_m)
#         else:
#             print( m )

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        #kaiming is first name of author whose last name is 'He' lol
        nn.init.kaiming_uniform(m.weight) 
        m.bias.data.zero_()

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially 
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def calculate_mean_iu(predictions, gts, num_classes):
    sum_iu = 0
    class_iu = []
    for i in xrange(num_classes):
        n_ii = t_i = sum_n_ji = 1e-9
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        sum_iu += float(n_ii) / (t_i + sum_n_ji - n_ii)
        class_iu.append(float(n_ii) / (t_i + sum_n_ji - n_ii))
    mean_iu = sum_iu / num_classes
    return mean_iu,class_iu


def calculate_lane_metrics(predictions, gts, num_classes):
    sum_iu = 0
    class_iu = []
    acc = []
    rec = []
    f1_m = []
    for i in xrange(num_classes):
        tp = fp = fn = 0.
        for p, gt in zip(predictions, gts):
            tp += np.sum(gt[p == i] == i)
            fp += np.sum( (gt[p == i] != i ))
            fn += np.sum(gt[p != i] == i)

        class_iu.append(tp / (tp + fp + fn + 1e-9))
        acc.append(tp/(tp+fp+1e-9))
        rec.append(tp/(tp+fn+1e-9))
        f1_m.append(2*acc[i]*rec[i]/(acc[i]+rec[i]+1e-9))
        sum_iu += tp / (tp + fp + fn + 1e-9)
    mean_iu = sum_iu / num_classes
    return {'miu':mean_iu,
            'ciu':class_iu
            },\
            {'acc':acc,
             'rec':rec,
             'f1_m':f1_m
            }


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=None, reduction='mean')

    def forward(self, inputs, targets):
        # pdb.set_trace()
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)

def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def rm_file(path_file):
    if os.path.exists(path_file):
        os.remove(path_file)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cfg.VIS.PALETTE_LABEL_COLORS)

    return new_mask

def scores(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    # acc = np.diag(hist).sum() / hist.sum()
    # acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # return acc, acc_cls, mean_iu, fwavacc
    # pdb.set_trace()
    return {'miu':mean_iu,
            'ciu':iu
            }

def acc(predictions, gts, num_classes=2):
    predictions = predictions.data.numpy().astype(np.int64)
    gts = gts.data.numpy().astype(np.int64)
    predictions[predictions>=0.5]=1
    predictions[predictions<0.5]=0

    t = predictions==gts
    acc = np.sum(t)/float((t.shape[0]*t.shape[1]))
    return acc


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def streaming_scores(hist,predictions, gts, num_classes):
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    # acc = np.diag(hist).sum() / hist.sum()
    # acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # return hist, acc, acc_cls, mean_iu, fwavacc
    return hist, \
        {'miu':mean_iu,
        'ciu':iu
        }