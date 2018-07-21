import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from IPython import display
import torch


def plot_batch(img_batch, transpose_channels=True, limit=4):
    if transpose_channels:
        img_batch = np.transpose(img_batch, (0, 2, 3, 1))
    n = min(limit, img_batch.shape[0])
    ncols = int(sqrt(n))
    nrows = n // ncols
    if nrows * ncols < n:
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols)
    ax = axes.ravel()
    for i in range(n):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img_batch[i])


class PlotLossCallback(object):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.fig = plt.figure()
        self.l_ax, self.img_ax = self.fig.subplots(1, 2)
        self.d_losses = []
        self.g_losses = []
        self.Z = torch.Tensor(1, self.generator.z_size).cuda()

    def __call__(self, g_loss, d_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        g_handle, = self.l_ax.plot(self.g_losses, 'r', label="Generator loss")
        d_handle, = self.l_ax.plot(self.d_losses, 'b', label="Discriminator loss")
        self.l_ax.legend(handles=[g_handle, d_handle])

        self.Z.uniform_(-1., 1.)
        G_sample = self.generator(self.Z)
        self.img_ax.get_xaxis().set_visible(False)
        self.img_ax.get_yaxis().set_visible(False)
        self.img_ax.imshow(np.transpose(G_sample.data.cpu().numpy(), (0, 2, 3, 1))[0])
        display.display(plt.gcf())
        display.clear_output(wait=True)
        #print('G_loss: ', g_loss, 'D_loss:', d_loss)
