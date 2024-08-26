import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

try:
    from .utils.cbam import cbam_block
    from .utils.aspp import aspp
except:
    from utils.cbam import cbam_block
    from utils.aspp import aspp

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(2, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1,x2,x3,x4


def resnet50():
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3])


class Decoder_seg(nn.Module):
    def __init__(self):
        super(Decoder_seg, self).__init__()
        self.decoder1_seg = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder2_seg = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder3_seg = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0),
        )

        # Common practise for initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x_s1 = self.decoder1_seg(x)
        x_s2 = self.decoder2_seg(x_s1)
        x_s3 = self.decoder3_seg(x_s2)
        return x_s1, x_s2, x_s3


class Decoder_dis(nn.Module):
    def __init__(self):
        super(Decoder_dis, self).__init__()
        self.decoder1_dis = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder2_dis = nn.Sequential(
            nn.ConvTranspose2d(1025, 512, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder3_dis = nn.Sequential(
            nn.ConvTranspose2d(513, 256, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder4_dis = nn.Sequential(
            nn.ConvTranspose2d(257, 128, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0),
        )

        # Common practise for initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, x1, x2, x3, x_s1, x_s2, x_s3):
        x = self.decoder1_dis(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.cat([x, F.interpolate(x_s1, size=x.shape[-2:])], dim=1)
        x = self.decoder2_dis(x)
        x = torch.cat([x, x2], dim=1)
        x = torch.cat([x, F.interpolate(x_s2, size=x.shape[-2:])], dim=1)
        x = self.decoder3_dis(x)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, F.interpolate(x_s3, size=x.shape[-2:])], dim=1)
        x = self.decoder4_dis(x)
        return x

class PCorrNet(nn.Module):
    def __init__(self):
        super(PCorrNet, self).__init__()
        self.encoder = resnet50()
        self.aspp = aspp(2048,128)
        self.cbam = cbam_block(2048)
        self.decoder_seg = Decoder_seg()
        self.decoder_dis = Decoder_dis()
        self.down_3 = nn.Conv2d(1024, 512, 1)
        self.down_2 = nn.Conv2d(512, 256, 1)
        self.down_1 = nn.Conv2d(256, 128, 1)

    def forward(self, x):
        x_conv1, x_conv2, x_conv3, x_conv4 = self.encoder(x)
        x_s1, x_s2, x_s3 = self.decoder_seg(self.aspp(x_conv4))
        x_s1_mask = x_s1[:, -1:]
        x_s2_mask = x_s2[:, -1:]
        x_s3_mask = x_s3[:, -1:]
        dis = self.decoder_dis(self.cbam(x_conv4), self.down_3(x_conv3), self.down_2(x_conv2),
                             self.down_1(x_conv1), x_s1_mask, x_s2_mask, x_s3_mask)

        return x_s3, dis


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    net = PCorrNet().to(device)
    params = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (params/1e6))

    net.eval()
    with torch.no_grad():
        x = torch.rand((1, 2, 512, 512)).to(device)
        seg, dis = net(x)
        time0 = time.time()
        for _ in range(100):
            seg, dis = net(x)
        print((time.time() - time0) / 100)
        print(x.shape, seg.shape, dis.shape)
