import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg
from ori_big import ori
from misc.utils import CrossEntropyLoss2d

class CrowdCounter(nn.Module):
    def __init__(self, ce_weights=None):
        super(CrowdCounter, self).__init__()        
        self.CCN = ori()
        
        if len(cfg.TRAIN.GPU_ID)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=cfg.TRAIN.GPU_ID).cuda()
        else:
            self.CCN=self.CCN.cuda()

        if ce_weights is not None:
            ce_weights = torch.Tensor(ce_weights)
            ce_weights = ce_weights.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        self.loss_bce_fn = nn.BCELoss(weight=ce_weights).cuda() # binary cross entropy loss
        self.loss_cel_fn = CrossEntropyLoss2d().cuda()
        
    @property
    def loss(self):
        return self.loss_mse + cfg.TRAIN.BCE_WIEGHT*self.cross_entropy \
                             + cfg.TRAIN.SEG_WIEGHT*self.loss_seg

    def f_loss(self):
        return self.loss_mse, self.cross_entropy, self.loss_seg
    
    def forward(self, img, gt_map, roi, gt_roi, gt_seg):                               
        density_map, density_cls_score,pred_seg = self.CCN(img,roi)

        #20 lines for each pic(20 rois)
        density_cls_prob = F.softmax(density_cls_score)
                            
        self.loss_mse, self.cross_entropy, self.loss_seg = \
                                self.build_loss(density_map, density_cls_prob, pred_seg, gt_map, gt_roi, gt_seg)
            
            
        return density_map, density_cls_score, pred_seg
    
    def build_loss(self, density_map, density_cls_score, pred_seg, gt_data, gt_cls_label,gt_seg):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        # pdb.set_trace()
        cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)
        loss_seg = self.loss_cel_fn(pred_seg, gt_seg)  
        return loss_mse, cross_entropy,loss_seg


    def test_forward(self, img, roi):                               
        density_map, density_cls_score,pred_seg = self.CCN(img,roi)            
            
        return density_map, density_cls_score, pred_seg        

