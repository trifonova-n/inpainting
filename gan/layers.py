import torch.nn as nn
from math import sqrt


def down_4(in_channels, out_channels, kernel_size=5):
    middle_channels = int(sqrt(in_channels *out_channels))
    model = nn.Sequential(
        down_2(in_channels, middle_channels, kernel_size=kernel_size, batch_norm=True),
        down_2(middle_channels, out_channels, kernel_size=kernel_size, batch_norm=True)
    )
    return model


def down_2(in_channels, out_channels, kernel_size=5, batch_norm=True, relu_negative_slope=0.2):
    padding = (kernel_size - 1)//2
    model = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
    ]
    if batch_norm:
        model.append(nn.BatchNorm2d(out_channels))
    model.append(nn.LeakyReLU(relu_negative_slope))
    return nn.Sequential(*model)


def up_2(in_channels, out_channels, kernel_size=5, batch_norm=True):
    padding = (kernel_size - 1) // 2
    model = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1)
    ]
    if batch_norm:
        model.append(nn.BatchNorm2d(out_channels))
    model.append(nn.ReLU())
    return nn.Sequential(*model)


def up_4(in_channels, out_channels, kernel_size=5):
    middle_channels = int(sqrt(in_channels *out_channels))
    model = nn.Sequential(
        up_2(in_channels, middle_channels, kernel_size=kernel_size, batch_norm=True),
        up_2(middle_channels, out_channels, kernel_size=kernel_size, batch_norm=True)
    )
    return model
