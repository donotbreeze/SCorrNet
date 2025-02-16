import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision
from Models.SCorrNet import SCorrNet


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SCorrNet().to(device)
    net.load_state_dict(torch.load('The weights of different DIC networks/weights_SCorrNet.pth', map_location=device))
    net.eval()

    ref_image = Image.open("Reference and deformed images/Sample_tensile experiment/img_00000.png").convert('L')
    tar_image = Image.open("Reference and deformed images/Sample_tensile experiment/img_00019.png").convert('L')
    ref_image = torchvision.transforms.ToTensor()(ref_image)
    tar_image = torchvision.transforms.ToTensor()(tar_image)
    image = torch.cat((ref_image, tar_image), dim=0).unsqueeze(dim=0).to(device)

    _, _, h, w = image.shape

    seg, dis = net(image.to(device))
    dis = dis[:,:,:h,:w].cpu().detach().numpy()*10
    mask = torch.argmax(seg[:, :, :h, :w].cpu(), dim=1)

    # dis = net(image.to(device))
    # dis = dis[:, :, :h, :w].cpu().detach().numpy() * 10

    plt.figure()
    plt.imshow(dis[0][0],cmap='jet',vmin=-10,vmax=10)
    plt.colorbar()
    plt.savefig("Results/smaple_tensile experiment_dis_x.png")

    plt.figure()
    plt.imshow(dis[0][1],cmap='jet',vmin=-10,vmax=10)
    plt.colorbar()
    plt.savefig("Results/smaple_tensile experiment_dis_y.png")

    plt.figure()
    plt.imshow(mask[0],cmap='gray',vmin=0,vmax=1)
    plt.savefig("Results/smaple_tensile experiment_mask.png")

