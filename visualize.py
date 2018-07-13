import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from IPython import display


def plot_batch(img_batch, transpose_channels=True):
    if transpose_channels:
        img_batch = np.transpose(img_batch, (0, 2, 3, 1))
    n = img_batch.shape[0]
    ncols = int(sqrt(n))
    nrows = n // ncols
    if nrows * ncols < n:
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols)
    ax = axes.ravel()
    for i in range(img_batch.shape[0]):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img_batch[i])


class PlotLossCallback(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.d_losses = []
        self.g_losses = []

    def __call__(self, g_loss, d_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        g_handle, = self.ax.plot(self.g_losses, 'r', label="Generator loss")
        d_handle, = self.ax.plot(self.d_losses, 'b', label="Discriminator loss")
        self.ax.legend(handles=[g_handle, d_handle])
        display.display(plt.gcf())
        display.clear_output(wait=True)
        #print('G_loss: ', g_loss, 'D_loss:', d_loss)
