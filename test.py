from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.ori_big import ori
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio

import heatmap

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './DULR-display-save-mat'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

mean_std = cfg.DATA.MEAN_STD
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

wts = torch.FloatTensor(
        [ 0.07259259,  0.05777778,  0.10148148,  0.10592593,  0.10925926,\
        0.11      ,  0.11037037,  0.11074074,  0.11111111,  0.11074074]    
            )
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = '/media/D/DataSet/CC/576x768/shanghaitech_part_A/test_data'

model_path = '/media/D/gcy/code/DULR/exp/ori_big_seg_02-01_15-31_ori_0.0001_0.01_0.0001_0.995_10/all_ep_1861.pth'

def main():
    # file_list = [filename for filename in os.listdir(dataRoot+'/img/') if os.path.isfile(os.path.join(dataRoot+'/img/',filename))]
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]
    # pdb.set_trace()                     

    ht_img = cfg.TRAIN.INPUT_SIZE[0]
    wd_img = cfg.TRAIN.INPUT_SIZE[1]        
    # gengrate roi info
    roi = torch.zeros((20,5))
    for i in range(0,20):
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

    # roi = roi.long()    

    roi = Variable(roi[None,:,:],volatile=True).cuda()                       

    test(file_list[0], model_path,roi)
   

def test(file_list, model_path,roi):

    net = CrowdCounter(ce_weights=wts)
    net.load_state_dict(torch.load(model_path))
    # net = tr_net.CNN()
    # net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    for filename in file_list:
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'


        den = pd.read_csv(denname, sep=',',header=None).as_matrix()
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        # prepare
        wd_1, ht_1 = img.size

        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            pad = np.zeros([ht_1,dif])
            img = np.array(img)
            den = np.array(den)
            img = np.hstack((img,pad))
            img = Image.fromarray(img.astype(np.uint8))
            den = np.hstack((den,pad))
            
        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            pad = np.zeros([dif,wd_1])
            img = np.array(img)
            den = np.array(den)
            # pdb.set_trace()
            img = np.vstack((img,pad))
            img = Image.fromarray(img.astype(np.uint8))

            den = np.vstack((den,pad))

        img = img_transform(img)



        gt = np.sum(den)
        # den = Image.fromarray(den)

        img = img*255.

        img = Variable(img[None,:,:,:],volatile=True).cuda()

        #forward
        pred_map,pred_cls,pred_seg = net.test_forward(img, roi)

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        pred = np.sum(pred_map)
        pred_map = pred_map/np.max(pred_map+1e-20)
        pred_map = pred_map[0:ht_1,0:wd_1]
        
        
        den = den/np.max(den+1e-20)
        den = den[0:ht_1,0:wd_1]

        den_frame = plt.gca()
        plt.imshow(den)
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False) 
        den_frame.spines['bottom'].set_visible(False) 
        den_frame.spines['left'].set_visible(False) 
        den_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()
        
        sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map)
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

        '''pdb.set_trace()

        pil_input = restore(img[0]/255.)
        pil_to_tensor(pil_input.convert('RGB'))

        pdb.set_trace()'''

def get_pts(data):
    pts = []
    cols,rows = data.shape
    data = data*100

    for i in range(0,rows):  
        for j in range(0,cols):  
            loc = [i,j]
            for i_pt in range(0,int(data[i][j])):
                pts.append(loc) 
    return pts               



if __name__ == '__main__':
    main()




