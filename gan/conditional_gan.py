import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from .layers import up_2, up_4, down_2, down_4
from gan.hyperparameters import CondGeneratorParams, CondDiscriminatorParams, Params


class GeneratorNet(torch.nn.Module):
    def __init__(self, params=None):
        super(GeneratorNet, self).__init__()
        if params is None:
            params = CondGeneratorParams()
        else:
            params = CondGeneratorParams(**params)

        self.hyper_params = params
        cond_channels = int(params.min_shape[0]*params.condition_to_noise)
        params.cond_shape = (cond_channels, *params.min_shape[1:])
        params.noise_shape = (params.min_shape[0] - cond_channels, *params.min_shape[1:])

        # layer for linear transformation of noise vector to feature map with noise_shape
        noise_transf_layer = [nn.Linear(params.z_size, int(np.prod(params.noise_shape)))]
        if params.bn_start_idx == 0:
            noise_transf_layer.append(nn.BatchNorm1d(int(np.prod(params.min_shape))))
        noise_transf_layer.append(nn.ReLU())
        self.noise_transf_layer = nn.Sequential(*noise_transf_layer)

        # layer for linear transformation of noise vector to feature map with cond_shape
        self.cond_transf_layer = nn.Sequential(
            nn.Linear(params.y_size, int(np.prod(params.cond_shape))),
            nn.ReLU()
        )

        # Compute number of upscale layers nesessary to generate image of shape out_shape
        n_up_layers = int(round(np.log2(params.out_shape[1] // params.min_shape[1])))
        n_layers = n_up_layers + 2
        if params.bn_end_idx < 0:
            params.bn_end_idx = n_layers + params.bn_end_idx

        # up_layer upscale image by 2 n_up_layers times
        up_layer = []
        in_channels = params.min_shape[0]
        for idx in range(1, n_up_layers + 1):
            out_channels = int(in_channels / params.channel_scaling_factor)
            use_batchnorm = params.bn_start_idx <= idx <= params.bn_end_idx
            up_layer.append(up_2(in_channels, out_channels, kernel_size=5, batch_norm=use_batchnorm))
            in_channels = out_channels
        self.up_layer = nn.Sequential(*up_layer)

        # out_layer generate image with 3 channels and values from -1 to 1
        out_layer = [nn.ConvTranspose2d(in_channels, params.out_shape[0], kernel_size=3, stride=1, padding=1)]
        if params.bn_end_idx >= n_layers - 1:
            out_layer.append(nn.BatchNorm2d(3))
        out_layer.append(nn.Tanh())
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, z, y):
        noise = self.noise_transf_layer(z).view(-1, *self.hyper_params.noise_shape)
        cond = self.cond_transf_layer(y).view(-1, *self.hyper_params.cond_shape)
        input = torch.cat((noise, cond), 1)
        out = self.up_layer(input)
        out = self.out_layer(out)
        return out


class DiscriminatorNet(torch.nn.Module):
    def __init__(self, params=None):
        super(DiscriminatorNet, self).__init__()
        if params is None:
            params = CondDiscriminatorParams()
        else:
            params = CondDiscriminatorParams(**params)
        self.hyper_params = params

        # compute number of downscale layers nesessary to transform image
        # to features of shape feature_img_size x feature_img_size
        n_down_layers = int(round(np.log2(params.input_shape[1] // params.feature_img_size)))
        n_layers = n_down_layers + 1
        if params.bn_end_idx < 0:
            params.bn_end_idx = n_layers + params.bn_end_idx

        params.cond_channels = int(params.start_channels*params.condition_to_image)
        # downscale image by 2 once before adding conditions
        self.down_precond_layer = down_2(params.input_shape[0],
                                         params.start_channels - params.cond_channels,
                                         batch_norm=(params.bn_start_idx == 0))

        # layer to transform condition vector y to feature vector of cond_channels size
        self.cond_transf_layer = nn.Sequential(
            nn.Linear(params.y_size, params.cond_channels),
            nn.ReLU()
        )
        # down_layer downscale image by 2 n_down_layers - 1 times
        down_layer = []
        in_channels = params.start_channels
        for idx in range(1, n_down_layers):
            out_channels = int(in_channels * params.channel_scaling_factor)
            use_batchnorm = params.bn_start_idx <= idx <= params.bn_end_idx
            down_layer.append(down_2(in_channels, out_channels, batch_norm=use_batchnorm))
            in_channels = out_channels
        self.down_layer = nn.Sequential(*down_layer)
        self.hyper_params.min_shape = (in_channels, params.feature_img_size, params.feature_img_size)
        dim = int(np.prod(self.hyper_params.min_shape))
        self.logit_layer = nn.Linear(dim, 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x, y):
        dim = int(np.prod(self.hyper_params.min_shape))
        out = self.down_precond_layer(x)
        cond = self.cond_transf_layer(y)
        # repeate cond vector across feature map width and height
        cond = cond.view(-1, self.hyper_params.cond_channels, 1, 1).expand(-1, -1, out.shape[2], out.shape[3])
        # combine features and conditions after first convolution layer
        out = torch.cat((out, cond), 1)
        out = self.down_layer(out)
        out = out.reshape(-1, dim)
        D_logit = self.logit_layer(out)
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

