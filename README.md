# PCC Net: Perspective Crowd Counting via Spatial Convolutional Network
This is an official implementation of the paper "PCC net" (PCC Net: Perspective Crowd Counting via Spatial Convolutional Network).

This is the PCC Net with the VGG-16 backbone, which has a more powerful capacity for crowd counting than the original model in the paper.

Different from the original model:
1) it needs 3-channel RGB-color image;
2) the kernel sizes of some layers are smaller;
3) remove the max-pooling layers in the backend.


On the Shanghai Tech Part B with the size of 576*768, it achieves the MAE of XXX and MSE of XXX. Using high-resolution data will further improve the counting performance.

