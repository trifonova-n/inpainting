import torch
import torch.nn as nn
from math import sqrt


class GeneratorNet(torch.nn.Module):
    def __init__(self, z_size, y_size):
        super().__init__()
        self.z_size = z_size
        self.y_size = y_size
        self.layer0 = nn.Linear(z_size + y_size, 1024*4*4)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.noise = torch.Tensor(64, 64).cuda()
        self.eps = 0.01

    def forward(self, z, y):
        """
        :param z: noise tensor (b, z_size)
        :param y: condition tensor (b, y_size)
        :return:
        """
        input = torch.cat((z, y), 1)
        out = self.layer0(input).view(-1, 1024, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class Discriminator(torch.nn.Module):
    def __init__(self, y_size):
        super().__init__()
        self.y_size = y_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        self.layer4 = nn.Linear(512*2*2 + self.y_size, 512)
        self.layer5 = nn.Linear(512, 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).view(-1, 512*2*2)
        out = torch.cat((out, y), 1)
        out = self.layer4(out)
        D_logit = self.layer5(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit


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
        self.layer0 = self.down_2(3, 16, batch_norm=False)
        # 16x32x32
        self.layer1 = self.down_2(16 + self.y_size, 64)
        # 64x16x16
        self.layer2 = self.down_4(64, 256)
        # 256x4x4
        self.layer3 = self.down_4(256, 1024, kernel_size=3)
        # 1024x1x1
        self.layer4 = nn.Linear(1024, 1)
        self.out_layer = nn.Sigmoid()

    def down_4(self, in_channels, out_channels, kernel_size=5):
        middle_channels = int(sqrt(in_channels*out_channels))
        padding = (kernel_size - 1)//2
        return nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(self.relu_negative_slope),
            nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(self.relu_negative_slope)
        )

    def down_2(self, in_channels, out_channels, kernel_size=5, batch_norm=True):
        padding = (kernel_size - 1)//2
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(self.relu_negative_slope),
                nn.BatchNorm2d(out_channels)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(self.relu_negative_slope)
            )

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

