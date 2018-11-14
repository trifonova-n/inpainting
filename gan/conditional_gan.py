import torch
import torch.nn as nn
from math import sqrt
from .layers import up_2, up_4, down_2, down_4


class Generator5Net(torch.nn.Module):
    def __init__(self, z_size, y_size):
        super().__init__()
        self.z_size = z_size
        self.y_size = y_size
        self.layer0z = nn.Linear(z_size, 512*4*4)
        self.layer0y = nn.Linear(y_size, 512*4*4)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.noise = torch.Tensor(64, 64).cuda()
        self.eps = 0.01

    def forward(self, z, y):
        z = self.layer0z(z).view(-1, 512, 4, 4)
        y = self.layer0y(y).view(-1, 512, 4, 4)
        input = torch.cat((z, y), 1)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Discriminator5(torch.nn.Module):
    def __init__(self, y_size, relu_negative_slope=0.2):
        super().__init__()
        self.y_size = y_size
        self.relu_negative_slope = relu_negative_slope
        # input 3x64x64
        self.layer0 = down_2(3, 16, batch_norm=False)
        # 16x32x32
        self.layer1 = down_2(16 + self.y_size, 64)
        # 64x16x16
        self.layer2 = down_4(64, 256)
        # 256x4x4
        self.layer3 = down_4(256, 1024, kernel_size=3)
        # 1024x1x1
        self.layer4 = nn.Linear(1024, 1)
        self.out_layer = nn.Sigmoid()



    def forward(self, x, y):
        out = self.layer0(x)
        # expand y to batch_size x y_size x 32 x 32
        y = y.view(-1, self.y_size, 1, 1).expand(-1, -1, out.shape[2], out.shape[3])
        out = torch.cat((out, y), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(-1, 1024)
        D_logit = self.layer4(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit

