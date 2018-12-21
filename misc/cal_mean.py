import argparse
import torchvision.datasets as dset



import pdb
from PIL import Image
import numpy as np
import os


# TODO 

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDataPath', type=str, default='/media/D/gcy/code/2017.11/crowdcount-mlt-detection-2.0/data/part_B_final/train_data/images', 
                        help='absolute path to your data path')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()

    imgs_list = []

    for i_img, img_name in enumerate(os.listdir(args.trainDataPath)):
        if i_img % 100 == 0:
            print i_img
        img = Image.open(os.path.join(args.trainDataPath, img_name))
        if img.mode == 'RGB':
            img = img.convert('L')

        img = np.array(img.resize((1024,768),Image.BILINEAR))

        imgs_list.append(img)

    imgs = np.array(imgs_list).astype(np.float32)/255.
    # red = imgs[:,:,:,0]
    # green = imgs[:,:,:,1]
    # blue = imgs[:,:,:,2]


    print("means: {}".format(np.mean(imgs)))
    print("stdevs: {}".format(np.std(imgs)))