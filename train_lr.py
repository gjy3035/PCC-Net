from tensorboardX import SummaryWriter
import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils


from models.CC import CrowdCounter
from config import cfg
from loading_data import loading_data
from misc.utils import *
from misc.timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)
log_txt = cfg.TRAIN.EXP_PATH + '/' + exp_name + '/' + exp_name + '.txt'


if not os.path.exists(cfg.TRAIN.EXP_PATH):
    os.mkdir(cfg.TRAIN.EXP_PATH)
    
pil_to_tensor = standard_transforms.ToTensor()

train_record = {'best_mae': 1e20, 'mse':1e20,'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}

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

    net = CrowdCounter(ce_weights=train_set.wts).cuda()
    
    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()       

    net.train()

    optimizer = optim.Adam([
                            {'params': [param for name, param in net.named_parameters() if 'seg' in name], 'lr': cfg.TRAIN.SEG_LR},
                            {'params': [param for name, param in net.named_parameters() if 'seg' not in name], 'lr': cfg.TRAIN.LR}
                          ])
    
    i_tb = 0
    for epoch in range(cfg.TRAIN.MAX_EPOCH):

        _t['train time'].tic()
        i_tb,model_path = train(train_loader, net, optimizer, epoch, i_tb)
        _t['train time'].toc(average=False)
        print 'train time of one epoch: {:.2f}s'.format(_t['train time'].diff)
        if epoch%cfg.VAL.FREQ!=0:
            continue
        _t['val time'].tic()
        validate(val_loader, model_path, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print 'val time of one epoch: {:.2f}s'.format(_t['val time'].diff)


def train(train_loader, net, optimizer, epoch, i_tb):
    
    for i, data in enumerate(train_loader, 0):
        _t['iter time'].tic()
        img, gt_map, gt_cnt, roi, gt_roi, gt_seg = data

        for i_img in range(cfg.TRAIN.BATCH_SIZE):
            roi[i_img,:,0] = i_img
        roi = roi.view(cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.NUM_BOX,5)
        gt_roi = gt_roi.view(cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.NUM_BOX,10)

        # print gt_roi
        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()
        roi = Variable(roi).cuda().float()
        gt_roi = Variable(gt_roi).cuda()
        gt_seg = Variable(gt_seg).cuda()

        optimizer.zero_grad()
        pred_map,pred_cls, pred_seg = net(img, gt_map, roi, gt_roi, gt_seg)

        loss = net.loss
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        if (i + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            
            loss1,loss2,loss3 = net.f_loss()

            i_tb = i_tb + 1
            writer.add_scalar('train_loss_mse', loss1.data[0], i_tb)
            writer.add_scalar('train_loss_cls', loss2.data[0], i_tb)
            writer.add_scalar('train_loss_seg', loss3.data[0], i_tb)
            writer.add_scalar('train_loss', loss.data[0], i_tb)

            _t['iter time'].toc(average=False)
            print '[ep %d][it %d][loss %.8f %.8f %.4f %.4f][%.2fs]' % \
                    (epoch + 1, i + 1, loss.data[0], loss1.data[0], loss2.data[0], loss3.data[0], _t['iter time'].diff)
            # pdb.set_trace()
            print '        [cnt: gt: %.1f pred: %.6f]' % (gt_cnt[0], pred_map[0,:,:,:].sum().data[0])              
    
    snapshot_name = 'all_ep_%d' % (epoch + 1)
    # save model
    to_saved_weight = []

    if len(cfg.TRAIN.GPU_ID)>1:
        to_saved_weight = net.module.state_dict()                
    else:
        to_saved_weight = net.state_dict()
    model_path = os.path.join(cfg.TRAIN.EXP_PATH, exp_name, snapshot_name + '.pth')
    torch.save(to_saved_weight, model_path)

    return i_tb,model_path

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
        gt_count = np.sum(gt_map)
        pred_cnt = np.sum(pred_map)

        mae += abs(gt_count-pred_cnt)
        mse += ((gt_count-pred_cnt)*(gt_count-pred_cnt))

        x = []
        if vi==0:
            for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, pred_seg, gt_seg)):
                if idx>cfg.VIS.VISIBLE_NUM_IMGS:
                    break
                # pdb.set_trace()
                pil_input = restore(tensor[0]/255.)
                pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
                pil_output = torch.from_numpy(tensor[1]/(tensor[1].max()+1e-10)).repeat(3,1,1)
                
                pil_gt_seg = torch.from_numpy(tensor[4]).repeat(3,1,1).float()
                pil_pred_seg = torch.from_numpy(tensor[3]).repeat(3,1,1).float()
                # pdb.set_trace()
                
                x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output, pil_gt_seg, pil_pred_seg])
            x = torch.stack(x, 0)
            x = vutils.make_grid(x, nrow=5, padding=5)
            writer.add_image(exp_name + '_epoch_' + str(epoch+1), (x.numpy()*255).astype(np.uint8))

    mae = mae/val_set.get_num_samples()
    mse = np.sqrt(mse/val_set.get_num_samples())

    '''
    loss1 = float(np.mean(np.array(val_loss_mse)))
    loss2 = float(np.mean(np.array(val_loss_cls)))
    loss3 = float(np.mean(np.array(val_loss_seg)))
    loss = float(np.mean(np.array(val_loss)))'''
    loss1 = np.mean(np.array(val_loss_mse))[0]
    loss2 = np.mean(np.array(val_loss_cls))[0]
    loss3 = np.mean(np.array(val_loss_seg))[0]
    loss = np.mean(np.array(val_loss))[0]    

    writer.add_scalar('val_loss_mse', loss1, epoch + 1)
    writer.add_scalar('val_loss_cls', loss2, epoch + 1)
    writer.add_scalar('val_loss_seg', loss3, epoch + 1)
    writer.add_scalar('val_loss', loss, epoch + 1)
    writer.add_scalar('mae', mae, epoch + 1)
    writer.add_scalar('mse', mse, epoch + 1)


    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_loss'] = loss        

    print '='*50
    print exp_name
    print '    '+ '-'*20
    print '    [mae %.1f mse %.1f], [val loss %.8f %.8f %.4f %.4f]' % (mae, mse, loss, loss1, loss2, loss3)         
    print '    '+ '-'*20
    # pdb.set_trace()
    print '[best] [mae %.1f mse %.1f], [loss %.8f], [epoch %d]' % (train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch'])
    print '='*50


if __name__ == '__main__':
    main()








