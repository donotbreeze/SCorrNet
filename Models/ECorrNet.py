import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
try:
    from CorrNet import CorrNet
    from ENet import ENet
except:
    from .CorrNet import CorrNet
    from .ENet import ENet


class ECorrNet(nn.Module):
    def __init__(self):
        super(ECorrNet, self).__init__()
        self.enet = ENet()
        self.corrnet = CorrNet()

    def forward(self, x):
        seg = self.enet(x[:,0:1])
        mask = torch.argmax(seg, dim=1).unsqueeze(dim=1)
        dis = self.corrnet(torch.cat([x, mask], dim=1))
        return seg, dis


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ECorrNet().to(device)

    params = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (params/1e6))

    net.eval()
    with torch.no_grad():
        x = torch.rand((1, 2, 512, 512)).to(device)
        seg, dis = net(x)
        time0 = time.time()
        for _ in range(100):
            seg, dis = net(x)
        print((time.time() - time0)/100)
        print(x.shape, seg.shape, dis.shape)
