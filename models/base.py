import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # m.bias.data.fill_(0.01)


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        dilation=1,
    )


def conv1x1(in_ch, out_ch, stride=2):
    return nn.Conv2d(
        in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False
    )


class BlurPool(nn.Module):
    def __init__(self, in_channels, size, padding):
        super(BlurPool, self).__init__()

        self.size = size
        self.in_channels = in_channels

        if self.size == 2:
            v = np.array([1.0, 1.0])
        elif self.size == 3:
            v = np.array([1.0, 2.0, 1.0])
        elif self.size == 5:
            v = np.array([1.0, 4.0, 6.0, 4.0, 1.0])

        w = torch.Tensor(v[:, None] * v[None, :])
        w = w / torch.sum(w)

        self.register_buffer(
            "W", w[None, None, :, :].repeat((self.in_channels, 1, 1, 1))
        )

        if padding == "reflect":
            PadLayer = nn.ReflectionPad2d
        elif padding == "replicate":
            PadLayer = nn.ReplicationPad2d
        elif padding == "zero":
            PadLayer = nn.ZeroPad2d
        else:
            print("Pad type [{:s}] not recognized".format(padding))

        self.pad = PadLayer(
            [
                int((size - 1) / 2),
                int((size - 1) / 2),
                int((size - 1) / 2),
                int((size - 1) / 2),
            ]
        )

    def forward(self, x):
        return F.conv2d(self.pad(x), self.W, stride=2, groups=self.in_channels)


def conv_block(in_chnl, out_chnl, pool=False, padding=1):
    layers = [
        nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_chnl),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_ch, in_dim, c_num):
        super().__init__()

        self.conv1 = conv_block(in_ch, in_dim, pool=True)
        self.conv2 = conv_block(64, 128, pool=True)
        self.resnet1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.resnet2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.linear = nn.Sequential(
            nn.MaxPool2d(3), nn.Flatten(), nn.Linear(512, c_num)
        )  # num_cls

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.resnet1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.resnet2(out) + out
        return self.linear(out)
