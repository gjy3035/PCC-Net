# PCC Net: Perspective Crowd Counting via Spatial Convolutional Network
This is an official implementation of the paper "PCC net" (PCC Net: Perspective Crowd Counting via Spatial Convolutional Network).

![PCC Net.](./imgs/img0.png "pcc")

In the paper, the experiments are conducted on the three populuar datasets: Shanghai Tech, UCF_CC_50 and WorldExpo'10. To be specific, Shanghai Tech Part B contains crowd images with the same resolution. For easier data prepareation, we only release the pre-trained model on ShanghaiTech Part B dataset in this repo.

## Bracnhes

1. [ori_pt0.2_py2](https://github.com/gjy3035/PCC-Net/tree/ori_pt0.2_py2): the original version.
2. [ori_pt1_py3](https://github.com/gjy3035/PCC-Net): current version.
3. [vgg_pt1_py3](https://github.com/gjy3035/PCC-Net/tree/vgg_pt1_py3): vgg-backbone PCC Net (higher performance).

##  Requirements
- Python 3.x
- Pytorch 1.x
- TensorboardX (pip)
- torchvision  (pip)
- easydict (pip)
- pandas  (pip)


## Data preparation
1. Download the original ShanghaiTech Dataset [Link: [Dropbox ](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)/ [BaiduNetdisk](https://pan.baidu.com/s/1nuAYslz)]
2. Resize the images and the locations of key points. 
3. Generate the density maps by using the [code](https://github.com/aachenhang/crowdcount-mcnn/tree/master/data_preparation).
4. Generate the segmentation maps.

We also provide the processed Part B dataset for training. [[Link](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/gjy3035_mail_nwpu_edu_cn/EcMLqr9XuH1ChAgkqpxL_6kBK9EyCmIuXMxTb09FrjMYow?e=LJnOcC)]

## Training model
1. Run the train_lr.py: ```python train_lr.py```.
2. See the training outputs: ```Tensorboard --logdir=exp --port=6006```.

In the experiments, training  and tesing 800 epoches take 21 hours on GTX 1080Ti. 

## Expermental results

### Quantitative results

We show the Tensorboard visualization results as below:
![Detialed infomation during the traning phase.](./imgs/img1.jpg "pcc_q")
The mae and mse are the results on test set. Others are triaining loss. 

### Visualization results
Visualization results on the test set as below:
![Visualization results on the test set.](./imgs/img2.jpg "pcc_v")
Column 1: input image; Column 2: density map GT; Column 3: density map prediction; Column 4: segmentation map GT; Column 5: segmentation map prediction.

## Citation
If you use the code, please cite the following paper:
