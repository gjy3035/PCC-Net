import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from torch.utils import data
from PIL import Image
from config import cfg

import pdb

# train info
min_gt_count = 2.7126654189217972e-05
max_gt_count = 0.001306603490202515

wts = torch.FloatTensor(
       [ 0.10194444,  0.07416667,  0.08361111,  0.09277778,  0.10388889,\
        0.10416667,  0.10805556,  0.11      ,  0.11111111,  0.11027778]    
            )
box_num = cfg.TRAIN.NUM_BOX

class SHT_B(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        #num_classes: total number of classes into which the crowd count is divided (default: 10 as used in the paper)
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.seg_path = data_path + '/seg'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform    

        self.min_gt_count = min_gt_count
        self.max_gt_count = max_gt_count
        self.num_classes = 10
        self.box_num = box_num
        
        self.wts = wts

        # self.count_class_hist = np.zeros(self.num_classes)
        # self.get_stats_in_dataset() #get min - max crowd count present in the dataset. used later for assigning crowd group/class       
        # self.wts = self.get_classifier_weights()
        # pdb.set_trace()
        self.bin_val = (self.max_gt_count - self.min_gt_count)/float(self.num_classes)
        
    
    def __getitem__(self, index):
        fname = self.data_files[index]

        img, den, seg = self.read_image_and_gt(fname)
      
        if self.main_transform is not None:
            img, den, seg = self.main_transform(img,den,seg) 

        if self.img_transform is not None:
            img = self.img_transform(img)

        img = img# *255.
        den = torch.from_numpy(np.array(den))*cfg.DATA.DEN_ENLARGE
        seg = torch.from_numpy(np.array(seg).astype(np.uint8)).long()
        gt_count = den.sum()    
  
        ht_img = cfg.TRAIN.INPUT_SIZE[0]
        wd_img = cfg.TRAIN.INPUT_SIZE[1]        
        # gengrate roi info
        roi = torch.zeros((self.box_num,5))
        roi_label = torch.zeros(self.box_num,self.num_classes)
        for i in range(0,self.box_num):
            ht = 0
            wd = 0
            while (ht < (ht_img/4) or wd < (wd_img/4)):
                xmin = random.randint(0,wd_img-2)
                ymin = random.randint(0,ht_img-2)
                xmax = random.randint(0,wd_img-2)
                ymax = random.randint(0,ht_img-2)                              
                wd = xmax - xmin
                ht = ymax - ymin

            roi[i][0] = int(0)
            roi[i][1] = int(xmin)
            roi[i][2] = int(ymin)
            roi[i][3] = int(xmax)
            roi[i][4] = int(ymax)

            pic_count = den[ymin:ymax-1,xmin:xmax-1].sum()
            class_idx = np.round(pic_count/(xmax-xmin)/(ymax-ymin)/self.bin_val)
            class_idx = int(min(class_idx,self.num_classes-1))
            roi_label[i][class_idx] = 1

        roi = roi.long()
            
        return img, den, gt_count, roi, roi_label, seg

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        seg = Image.open(os.path.join(self.seg_path,fname.split('.')[0]+'.png'))
        if img.mode == 'L':
            img = img.convert('RGB')
        wd_1, ht_1 = img.size

        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        den = den.astype(np.float32, copy=False)
        gt_count = torch.from_numpy(den).sum()

        # add padding

        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            pad = np.zeros([ht_1,dif])
            img = np.hstack((np.array(img),pad))
            seg = np.hstack((np.array(seg),pad))
            den = np.hstack((np.array(den),pad))

            img = Image.fromarray(img.astype(np.uint8))
            seg = Image.fromarray(seg.astype(np.uint8))
            
        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            pad = np.zeros([dif,wd_1])
            img = np.vstack((np.array(img),pad))
            seg = np.vstack((np.array(seg),pad))
            den = np.vstack((np.array(den),pad))

            img = Image.fromarray(img.astype(np.uint8))
            seg = Image.fromarray(seg.astype(np.uint8))
            
        den = Image.fromarray(den)
        return img, den, seg



    def get_classifier_weights(self):
        wts = self.count_class_hist
        wts = 1-wts/(sum(wts));
        wts = wts/sum(wts);
        return wts      
        
    
    def get_stats_in_dataset(self):
        
        min_count = sys.maxint
        max_count = 0
        gt_count_array = np.zeros(self.num_samples)
        i = 0
        for fname in self.data_files:
            den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values                      
            den  = den.astype(np.float32, copy=False)
            gt_count = np.sum(den)

            # pdb.set_trace()

            min_count = min(min_count, (gt_count/den.shape[0]/den.shape[1]))
            max_count = max(max_count, (gt_count/den.shape[0]/den.shape[1]))
            gt_count_array[i] = gt_count/den.shape[0]/den.shape[1]
            i+=1
        
        self.min_gt_count = min_count
        self.max_gt_count = max_count        
        bin_val = (self.max_gt_count - self.min_gt_count)/float(self.num_classes)
        # pdb.set_trace()
        class_idx_array = np.round(gt_count_array/bin_val)
        
        
        for class_idx in class_idx_array:
            class_idx = int(min(class_idx, self.num_classes-1))
            
            # pdb.set_trace()
            self.count_class_hist[class_idx]+=1
                


    def get_num_samples(self):
        return self.num_samples       
            
        