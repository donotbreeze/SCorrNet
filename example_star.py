import torch
import torchvision
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from Models.SCorrNet import SCorrNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SCorrNet().to(device)
    net.load_state_dict(torch.load('The weights of different DIC networks/weights_SCorrNet.pth', map_location=device))
    net.eval()

    ref_image = Image.open("Reference and deformed images/Sample_Star/Star-8-200-10_ref.tif").convert('L')
    tar_image = Image.open("Reference and deformed images/Sample_Star/Star-8-200-10_tar.tif").convert('L')
    ref_image = torchvision.transforms.ToTensor()(ref_image)
    tar_image = torchvision.transforms.ToTensor()(tar_image)
    image = torch.cat((ref_image, tar_image), dim=0).unsqueeze(dim=0)

    GT = np.load("Reference and deformed images/Sample_Star/Star-8-200-10_dis.npy").transpose(2, 0, 1)

    _, _, h, w = image.shape
    image = F.pad(image, pad=(0, 32 - w % 32, 0, 32 - h % 32), mode='constant', value=0)

    seg, dis = net(image.to(device))
    dis = dis[:, :, :h, :w].cpu().detach().numpy() * 10
    mask = torch.argmax(seg[:, :, :h, :w].cpu(), dim=1)

    # dis = net(image.to(device))
    # dis = dis[:, :, :h, :w].cpu().detach().numpy() * 10


    plt.figure()
    plt.imshow(dis[0][1], cmap='jet', vmin=-10, vmax=10)
    plt.colorbar()
    plt.savefig("Results/smaple_Star.png", bbox_inches='tight', dpi=600)
    plt.close()

    error = GT - dis[0]
    plt.imshow(error[1], cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig("Results/smaple_Star_error.png", bbox_inches='tight', dpi=600)

