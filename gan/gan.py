import torch
import torch.nn as nn
from .layers import up_2, up_4, down_2, down_4


class GeneratorNet(torch.nn.Module):
    def __init__(self, z_size):
        super(GeneratorNet, self).__init__()
        self.start_channels = 1024
        self.preout_channels = 64
        self.z_size = z_size
        self.layer0 = nn.Linear(z_size, self.start_channels*4*4)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(self.start_channels),
            nn.ReLU())
        up_layer = []
        in_channels = self.start_channels
        out_channels = self.start_channels // 2
        while out_channels >= self.preout_channels:
            up_layer.append(up_2(in_channels, out_channels, kernel_size=5, batch_norm=True))
            in_channels = out_channels
            out_channels = in_channels // 2
        self.up_layer = nn.Sequential(*up_layer)
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, z):
        out = self.layer0(z).view(-1, self.start_channels, 4, 4)
        out = self.layer1(out)
        out = self.up_layer(out)
        out = self.out_layer(out)
        return out


class Generator5Net(torch.nn.Module):
    def __init__(self, z_size):
        super(Generator5Net, self).__init__()
        self.z_size = z_size
        self.layer0 = nn.Linear(z_size, 1024*4*4)
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

    def forward(self, z):
        out = self.layer0(z).view(-1, 1024, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Discriminator5(torch.nn.Module):
    def __init__(self):
        super(Discriminator5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        self.layer4 = nn.Linear(512*2*2, 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).view(-1, 512*2*2)
        D_logit = self.layer4(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit

