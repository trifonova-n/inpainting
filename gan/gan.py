import torch
import torch.nn as nn
import numpy as np
from .layers import up_2, up_4, down_2, down_4


class Params(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class GeneratorParams(Params):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_shape = self.get('min_shape', (1024, 4, 4))
        self.channel_scaling_factor = self.get('channel_scaling_factor', 2)
        self.out_shape = self.get('out_shape', (3, 64, 64))
        # we don't add batchnorm layers in several first
        # and end layers
        # index of first layer with batchnorm
        self.bn_start_idx = self.get('bn_start_idx', 1)
        # index of last layer to have batchnorm
        self.bn_end_idx = self.get('bn_end_idx', -1)
        self.z_size = self.get('z_size', 100)


class DiscriminatorParams(Params):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = self.get('input_shape', (3, 64, 64))
        self.start_channels = self.get('start_channels', 32)
        self.channel_scaling_factor = self.get('channel_scaling_factor', 2)
        self.feature_img_size = self.get('feature_img_size', 2)
        # we don't add batchnorm layers in several first
        # and end layers
        # index of first layer with batchnorm
        self.bn_start_idx = self.get('bn_start_idx', 1)
        # index of last layer to have batchnorm
        self.bn_end_idx = self.get('bn_end_idx', -1)


class GeneratorNet(torch.nn.Module):
    def __init__(self, z_size, params=None):
        super(GeneratorNet, self).__init__()
        if params is None:
            params = GeneratorParams()
        else:
            params = GeneratorParams(**params)

        self.hyper_params = params

        # layer for linear transformation of noise vector to feature map with min_shape
        noise_transf_layer = [nn.Linear(z_size, np.prod(params.min_shape))]
        if params.bn_start_idx == 0:
            noise_transf_layer.append(nn.BatchNorm1d(np.prod(params.min_shape)))
        noise_transf_layer.append(nn.ReLU())
        self.noise_transf_layer = nn.Sequential(*noise_transf_layer)

        n_up_layers = int(round(np.log2(params.out_shape[1] // params.min_shape[1])))
        n_layers = n_up_layers + 2
        if params.bn_end_idx < 0:
            params.bn_end_idx = n_layers + params.bn_end_idx

        # up_layer upscale image by 2 n_up_layers times
        up_layer = []
        in_channels = params.min_shape[0]
        for idx in range(1, n_up_layers + 1):
            out_channels = in_channels // self.channel_scaling_factor
            use_batchnorm = params.bn_start_idx <= idx < params.bn_end_idx
            up_layer.append(up_2(in_channels, out_channels, kernel_size=5, batch_norm=use_batchnorm))
            in_channels = out_channels
        self.up_layer = nn.Sequential(*up_layer)

        # out_layer generate image with 3 channels and values from -1 to 1
        out_layer = [nn.ConvTranspose2d(in_channels, params.out_shape[0], kernel_size=3, stride=1, padding=1)]
        if params.bn_end_idx >= n_layers - 1:
            out_layer.append(out_layer.append(nn.BatchNorm2d(3)))
        out_layer.append(nn.Tanh())
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, z):
        out = self.noise_transf_layer(z).view(-1, *self.hyper_params.min_shape)
        out = self.up_layer(out)
        out = self.out_layer(out)
        return out


class DiscriminatorNet(torch.nn.Module):
    def __init__(self, params=None):
        super(DiscriminatorNet, self).__init__()
        if params is None:
            params = DiscriminatorParams()
        else:
            params = DiscriminatorParams(**params)

        self.hyper_params = params

        n_down_layers = int(round(np.log2(params.input_shape[1] // params.feature_img_size)))
        n_layers = n_down_layers + 1
        if params.bn_end_idx < 0:
            params.bn_end_idx = n_layers + params.bn_end_idx

        down_layer = [
                down_2(params.input_shape[0], params.start_channels, batch_norm=(params.bn_start_idx == 0))
            ]

        in_channels = params.start_channels
        for idx in range(1, n_down_layers):
            out_channels = in_channels * params.channel_scaling_factor
            use_batchnorm = params.bn_start_idx <= idx < params.bn_end_idx
            down_layer.append(down_2(in_channels, out_channels, batch_norm=use_batchnorm))
            in_channels = out_channels
        self.down_layer = nn.Sequential(*down_layer)
        self.hyper_params.min_shape = (in_channels, params.feature_img_size, params.feature_img_size)
        self.logit_layer = nn.Linear(np.prod(self.hyper_params.min_shape), 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.down_layer(x).view(-1, np.prod(self.hyper_params.min_shape))
        D_logit = self.logit_layer(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit


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

