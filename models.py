
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0., 0.02)
        #m.bias.data.fill_(0.01)       

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                    stride=stride, padding=1, bias=False, dilation=1)

def conv1x1(in_ch, out_ch, stride=2):
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                    stride=stride, bias=False)

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResNetBlock, self).__init__()

        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_ch, out_ch, stride),
                nn.BatchNorm2d(out_ch)
            )

        self.block = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_ch, out_ch, stride)),
            ('bn1'  , nn.BatchNorm2d(out_ch)),
            ('relu' , nn.ReLU(inplace=True)),
            ('conv2', conv3x3(out_ch, out_ch)),
            ('bn2'  , nn.BatchNorm2d(out_ch)),
        ]))

    def forward(self, x):
        out  = self.block(x)
        out += (x if self.downsample is None else self.downsample(x))
        out  = F.relu(out)

        return out

class BaseModel(nn.Module):
    def __init__(self, in_ch, in_dim, c_num):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )                                                    # 32x48x48

        self.conv2 = ResNetBlock(in_dim * 1, in_dim * 2, 2)  # 64x24x24
        self.conv3 = ResNetBlock(in_dim * 2, in_dim * 4, 2)  # 128x12x12
        self.conv4 = ResNetBlock(in_dim * 4, in_dim * 4, 2)  # 128x6x6
        self.conv5 = ResNetBlock(in_dim * 4, in_dim * 4, 2)  # 128x3x3
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc = nn.Linear(in_dim * 4, c_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)

        return x

