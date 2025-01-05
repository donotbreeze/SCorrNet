# A loosely coupled serial digital image correlation method based on deep learning

This code serves as a supporting code for the article “A loosely coupled serial digital image correlation method based on deep learning”.

In deformation measurement, the measurement object has various shapes, and the measurement image inevitably contains background. Traditional DIC methods need to complete the complex process of speckle region selection, subset size setting and grid step setting according to the object shape, while the existing deep learning DIC methods are difficult to cope with this situation, especially the dynamic background. 
In order to address the above issues, a DIC neural network SCorrNet is designed that cascades the segmentation network ENet and the displacement computation network CorrNet for the first time, which first segments the speckle region so that the subsequent network can focuses on the displacement computation of the speckle region. To accomplish the training and evaluation of this network, a self-built dataset containing three types of displacement fields, speckle region mask, reference image and deformed target image is constructed. Meanwhile, based on CorrNet, a parallel structure network PCorrNet is constructed as a comparison network to verify the effectiveness of SCorrNet.

The traditional DIC method and the deep learning methods StrainNet, DeepDIC, DICNet, PCorrNet, SCorrNet are involved in the comparison experiments.

## Dependencies

SCorrNet is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction.

- Python==3.9
- PyTorch==1.10.0
- Torchvision==0.11.0
- Pillow==9.3.0
- numpy==1.23.5
- CUDA==11.1

## Reference and deformed images

The file records the reference images and deformed images of each sample.

## Results

The file records the results of each sample.

## The weights of different DIC networks

Before running the code, you need to extract the compressed package in the weights folder firstly.

## Datasets

The self-built dataset is available on Microsoft OneDrive:

https://1drv.ms/f/s!Ag105hW9KtLsnkYd3mQ0yHYm8_gP?e=TSsj0G

If you are in China, the dataset can also be downloaded via Baidu Webdisk:

https://pan.baidu.com/s/14HvWHtRXKaVkLBgwNj47gA?pwd=bx91

DeepDIC dataset:

https://github.com/RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation

DIC challenge dataset:

https://www.idics.org/challenge/
