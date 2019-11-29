import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC, convDU, convLR

from torchvision.ops import RoIPool
from torchvision import models
import pdb
from misc.utils import weights_normal_init, initialize_weights

class ori(nn.Module):

    def __init__(self, bn=False, num_classes=10):
        super(ori, self).__init__()

        vgg = models.vgg16(pretrained=True)

        features = list(vgg.features.children())
        self.base_layer = nn.Sequential(*features[0:23])

        self.squz_layer = Conv2d(512, 256, 1, same_padding=True, NL='prelu', bn=bn)
        
        self.num_classes = num_classes        

        
        self.hl_prior = nn.Sequential(Conv2d( 256, 128, 3, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(128, 64, 3, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(64, 32, 3, same_padding=True, NL='prelu', bn=bn))
                
        self.roi_pool = RoIPool([16, 16], 1/8.0)
        self.hl_prior_conv2d = Conv2d( 32, 16, 1, same_padding=True, NL='prelu', bn=bn)
        
        self.bbx_pred = nn.Sequential(
                                    FC(16*16*16,512, NL='prelu'),
                                    FC(512,256,  NL='prelu'),
                                    FC(256, self.num_classes, NL='prelu')
                                    )
             
        # generate dense map
        self.den_stage_1 = nn.Sequential(Conv2d( 256, 128, 7, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(128, 64, 5, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(64, 32, 5, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(32, 32, 5, same_padding=True, NL='prelu', bn=bn))
        
        self.den_stage_DULR = nn.Sequential(convDU(in_out_channels=32,kernel_size=(1,9)),
                                        convLR(in_out_channels=32,kernel_size=(9,1)))


        self.den_stage_2 = nn.Sequential(Conv2d( 64, 64, 3, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 64, 32, 3, same_padding=True, NL='prelu', bn=bn),                                        
                                        nn.ConvTranspose2d(32,32,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16,8,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU())

        # generrate seg map
        self.seg_stage = nn.Sequential(Conv2d( 32, 32, 1, same_padding=True, NL='prelu', bn=bn),
        	                            Conv2d( 32, 64, 3, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 64, 32, 3, same_padding=True, NL='prelu', bn=bn),
                                        nn.ConvTranspose2d(32,32,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),                                        
                                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16,8,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU())

        self.seg_pred = Conv2d(8, 2, 1, same_padding=True, NL='relu', bn=bn)

        self.trans_den = Conv2d(8, 8, 1, same_padding=True, NL='relu', bn=bn)

        self.den_pred = Conv2d(16, 1, 1, same_padding=True, NL='relu', bn=bn)

        weights_normal_init(self.squz_layer, self.hl_prior, self.hl_prior_conv2d, self.bbx_pred, self.den_stage_1, \
                            self.den_stage_DULR, self.den_stage_2, self.trans_den, self.den_pred)

        initialize_weights(self.seg_stage,  self.seg_pred)

        
    def forward(self, im_data, roi):
        x_base = self.base_layer(im_data)
        x_base = self.squz_layer(x_base)

        x_hlp = self.hl_prior(x_base)
        x_bbx = self.roi_pool(x_hlp, roi)
        x_bbx = self.hl_prior_conv2d(x_bbx)

        x_bbx = x_bbx.view(x_bbx.size(0), -1) 
        x_bbx = self.bbx_pred(x_bbx)

        x_map_1 = self.den_stage_1(x_base)
        x_map_1 = self.den_stage_DULR(x_map_1) 

        x_map = torch.cat((x_hlp,x_map_1),1)
        x_map = self.den_stage_2(x_map)

        x_seg_fea = self.seg_stage(x_map_1)
        x_seg = self.seg_pred(x_seg_fea)

        x_seg_fea = self.trans_den(x_seg_fea)

        x_map = self.den_pred(torch.cat((x_map,x_seg_fea),1))



        return x_map, x_bbx, x_seg