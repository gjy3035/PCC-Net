import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from models.CC import CrowdCounter
from config import cfg
from loading_data import loading_data
from misc.utils import *
from misc.timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)
log_txt = cfg.TRAIN.EXP_PATH + '/' + exp_name + '/' + exp_name + '.txt'

pil_to_tensor = standard_transforms.ToTensor()

train_set, train_loader, val_set, val_loader, restore_transform = loading_data()

_t = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

rand_seed = cfg.TRAIN.SEED    
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    model_path = '/media/D/gcy/code/DULR/exp/ori_big_seg_02-01_15-31_ori_0.0001_0.01_0.0001_0.995_10/all_ep_1861.pth'

    validate(val_loader, model_path, 1, restore_transform)





def validate(val_loader, model_path, epoch, restore):
    net = CrowdCounter(ce_weights=train_set.wts)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    print '='*50
    val_loss_mse = []
    val_loss_cls = []
    val_loss_seg = []
    val_loss = []
    mae = 0.0
    mse = 0.0

    for vi, data in enumerate(val_loader, 0):
        img, gt_map, gt_cnt, roi, gt_roi, gt_seg = data
        # pdb.set_trace()
        img = Variable(img, volatile=True).cuda()
        gt_map = Variable(gt_map, volatile=True).cuda()
        gt_seg = Variable(gt_seg, volatile=True).cuda()

        roi = Variable(roi[0], volatile=True).cuda().float()
        gt_roi = Variable(gt_roi[0], volatile=True).cuda()

        pred_map,pred_cls,pred_seg = net(img, gt_map, roi, gt_roi, gt_seg)
        loss1,loss2,loss3 = net.f_loss()
        val_loss_mse.append(loss1.data)
        val_loss_cls.append(loss2.data)
        val_loss_seg.append(loss3.data)
        val_loss.append(net.loss.data)


        pred_map = pred_map.data.cpu().numpy()
        gt_map = gt_map.data.cpu().numpy()

        pred_seg = pred_seg.cpu().max(1)[1].squeeze_(1).data.numpy()
        gt_seg = gt_seg.data.cpu().numpy()

        # pdb.set_trace()
        # pred_map = pred_map*pred_seg

        gt_count = np.sum(gt_map)
        pred_cnt = np.sum(pred_map)

        mae += abs(gt_count-pred_cnt)
        mse += ((gt_count-pred_cnt)*(gt_count-pred_cnt))




    # pdb.set_trace()
    mae = mae/val_set.get_num_samples()
    mse = np.sqrt(mse/val_set.get_num_samples())

    loss1 = np.mean(np.array(val_loss_mse))[0]
    loss2 = np.mean(np.array(val_loss_cls))[0]
    loss3 = np.mean(np.array(val_loss_seg))[0]
    loss = np.mean(np.array(val_loss))[0]


    print '='*50
    print exp_name
    print '    '+ '-'*20
    print '    [mae %.1f mse %.1f], [val loss %.8f %.8f %.4f %.4f]' % (mae, mse, loss, loss1, loss2, loss3)         
    print '    '+ '-'*20    
    print '='*50


if __name__ == '__main__':
    main()








