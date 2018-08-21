import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from IPython import display
import torch


def plot_batch(img_batch, transpose_channels=True, normalize=False, limit=4, ax=None):
    if transpose_channels:
        img_batch = np.transpose(img_batch, (0, 2, 3, 1))
    if normalize:
        img_batch = (img_batch + 1.0)/2
    n = min(limit, img_batch.shape[0])
    ncols = int(sqrt(n))
    nrows = n // ncols
    if nrows * ncols < n:
        nrows += 1

    if ax is None:
        fig, axes = plt.subplots(nrows, ncols)
        ax = axes.ravel()
    for i in range(n):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img_batch[i])


class GanPlotLossCallback(object):
    def __init__(self, generator, discriminator, fig_name='gan.png'):
        self.generator = generator
        self.discriminator = discriminator
        self.fig = plt.figure()
        self.l_ax, self.img_ax = self.fig.subplots(1, 2)
        self.d_losses = []
        self.g_losses = []
        self.Z = torch.Tensor(1, self.generator.z_size).cuda()
        self.fig_name = fig_name

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
        self.img_ax.imshow(np.transpose((G_sample.data.cpu().numpy() + 1)/2, (0, 2, 3, 1))[0])
        plt.savefig(self.fig_name)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        #print('G_loss: ', g_loss, 'D_loss:', d_loss)


class cGanPlotLossCallback(object):
    def __init__(self, generator, discriminator, fig_name='conditional_gan.png'):
        self.generator = generator
        self.discriminator = discriminator
        self.fig = plt.figure()
        self.l_ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        img_ax0 = plt.subplot2grid((2, 4), (0, 2))
        img_ax1 = plt.subplot2grid((2, 4), (0, 3))
        img_ax2 = plt.subplot2grid((2, 4), (1, 2))
        img_ax3 = plt.subplot2grid((2, 4), (1, 3))
        self.img_ax = [img_ax0, img_ax1, img_ax2, img_ax3]
        self.d_losses = []
        self.g_losses = []
        self.Z = torch.Tensor(4, self.generator.z_size).cuda()
        # 'Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat'
        self.Y = torch.tensor([
                          [1., 0, 0, 0, 0],
                          [0., 0, 1, 0, 0],
                          [1., 1, 1, 0, 0],
                          [0., 1, 0, 1, 0]]).cuda()
        self.fig_name = fig_name

    def __call__(self, g_loss, d_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        g_handle, = self.l_ax.plot(self.g_losses, 'r', label="Generator loss")
        d_handle, = self.l_ax.plot(self.d_losses, 'b', label="Discriminator loss")
        self.l_ax.legend(handles=[g_handle, d_handle])

        self.Z.uniform_(-1., 1.)
        G_sample = self.generator(self.Z, self.Y)
        plot_batch(G_sample.data.cpu().numpy(), normalize=True, ax=self.img_ax)
        plt.savefig(self.fig_name)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        #print('G_loss: ', g_loss, 'D_loss:', d_loss)
