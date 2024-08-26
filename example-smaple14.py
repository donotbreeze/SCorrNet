from Models.SCorrNet import SCorrNet
import torch
import torchvision
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    net = SCorrNet().to(device)
    net.load_state_dict(torch.load('weights/weights_SCorrNet.pth', map_location=device))
    net.eval()


    ref_image = Image.open("Sample12/oht_cfrp_00.tiff").convert('L')
    tar_image = Image.open("Sample12/oht_cfrp_08.tiff").convert('L')
    ref_image = torchvision.transforms.ToTensor()(ref_image)
    tar_image = torchvision.transforms.ToTensor()(tar_image)
    image = torch.cat((ref_image, tar_image), dim=0).unsqueeze(dim=0)

    _,_,h,w = image.shape
    image = F.pad(image,pad=(0,32-w%32,0,32-h%32), mode='constant',value= 0)
    seg, dis = net(image.to(device))
    dis = dis[:,:,:h,:w].cpu().detach().numpy()*10
    mask = torch.argmax(seg[:,:,:h,:w].cpu(),dim=1)

    plt.figure()
    plt.imshow(dis[0][0],cmap='jet',vmin=dis[0][0].min(),vmax=dis[0][0].max())
    plt.colorbar()
    plt.savefig("Results/smaple12-dis_x.png")

    plt.figure()
    plt.imshow(dis[0][1],cmap='jet',vmin=dis[0][1].min(),vmax=dis[0][1].max())
    plt.colorbar()
    plt.savefig("Results/smaple12-dis_y.png")

    plt.figure()
    plt.imshow(mask[0],cmap='gray',vmin=0,vmax=1)
    plt.savefig("Results/smaple12-mask.png")


