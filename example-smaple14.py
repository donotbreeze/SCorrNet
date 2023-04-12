from Models.ECorrNet import ECorrNet
import torch
import torchvision
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    net = ECorrNet().to(device)
    net.load_state_dict(torch.load('weights/weights.pth', map_location=device))
    net.eval()


    ref_image = Image.open("Sample14/Sample14 Reference.tif").convert('L')
    tar_image = Image.open("Sample14/Sample14 L3 Amp0.1.tif").convert('L')
    ref_image = torchvision.transforms.ToTensor()(ref_image)
    tar_image = torchvision.transforms.ToTensor()(tar_image)[:,:ref_image.shape[-2],:ref_image.shape[-1]]
    image = torch.cat((ref_image, tar_image), dim=0).unsqueeze(dim=0)

    _,_,h,w = image.shape
    image = F.pad(image,pad=(0,32-w%32,0,32-h%32), mode='constant',value= 0)
    seg, dis = net(image.to(device))
    dis = dis[:,:,:h,:w].cpu().detach().numpy()*10
    mask = torch.argmax(seg[:,:,:h,:w].cpu(),dim=1)

    plt.figure()
    plt.imshow(dis[0][0],cmap='jet',vmin=-0.15,vmax=0.15)
    plt.colorbar()
    plt.savefig("smaple14-dis_x.png")

    plt.figure()
    plt.imshow(mask[0],cmap='gray',vmin=0,vmax=1)
    plt.savefig("smaple14-mask.png")


