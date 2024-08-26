# A loosely coupled serial digital image correlation method based on deep learning

Digital image correlation (DIC) is a non-contact optical measurement method of displacement field, which has been widely used in various fields. Traditional DIC method needs to manually complete the tedious calculation process such as speckle region selection and subset setting to cope with the dynamic background. To improve the measurement accuracy and practicability, a complex and diverse dataset with simulated speckle images, real speckle images, background and large deformation is proposed. Meanwhile, a loosely coupled serial measurement network SCorrNet is proposed to automatically segment speckle regions and measure its displacement fields, so as to eliminate the influence of dynamic background. Finally, the proposed method is compared with traditional and learning-based methods, including self-constructed parallel deep learning method, on self-built and public datasets, and validated in two practical experiments. The experimental results show that SCorrNet has high accuracy, large deformation measurement ability, and speckle segmentation capability to reduce the influence of dynamic background. Therefore, our method has high practical value, while the code has been open-sourced.

## Dependencies

SCorrNet is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction.

- Python==3.9
- PyTorch==1.10.0
- Torchvision==0.11.0
- Pillow==9.3.0
- numpy==1.23.5
- CUDA==11.1

## Weight

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
