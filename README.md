# ECorrNet: Loosely coupled two-stage digital image correlation based on deep learning

Digital image correlation (DIC) is a non-contact global optical measurement method of displacement field, which has been widely used in experimental mechanics. The traditional DIC has an inefficient and tedious experimental process and is difficult to achieve real-time measurement. While DIC based on deep learning has been gradually proposed and greatly improved the calculation speed, the network model design and dataset of these methods are not perfect. In order to simplify the experimental process and improve the computational efficiency, a new dataset with background is proposed, which contains real speckle images, simulated speckle images and three different types of displacement fields, and a loosely coupled two-stage measurement network ECorrNet is designed to automatically segment speckle regions and measure displacement fields. The proposed method is compared with traditional method and existing deep learning methods on self-built dataset and public datasets. The experimental results show that ECorrNet has high accuracy, stability and generalization performance on all datasets, and the ability of segmenting speckle regions makes ECorrNet have higher practical application value.

## Dependencies

ECorrNet is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction.

- Python==3.9
- PyTorch==1.10.0
- Torchvision==0.11.0
- Pillow==9.3.0
- numpy==1.23.5
- CUDA==11.1

## Datasets

Self-built dataset:

The Self-built dataset link will be given later.

DeepDIC dataset:

https://github.com/RuYangNU/Deep-Dic-deep-learning-based-digital-image-correlation

DIC challenge dataset:

https://www.idics.org/challenge/