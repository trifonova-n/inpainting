import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from IPython import display
import torch


def plot_batch(img_batch, transpose_channels=True, normalize=False, limit=4, ax=None, descriptions=None):
    """

    :param img_batch: numpy array of shape (batch_size, channels, height, width) if transpose_channels=True
                      or (batch_size, height, width, channels) if transpose_channels=False
    :param bool transpose_channels:
    :param bool normalize: if image pixel values are from -1 to 1 scale them to [0, 1] range
    :param limit: max number of images on plot
    :param ax: axes list to plot images on
    :param descriptions: list of text descriptions of the batch_size length
    """
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
        fig.tight_layout()
        try:
            ax = axes.ravel()
        except AttributeError as e:
            ax = [axes]
    for i in range(n):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img_batch[i])
        if descriptions:
            ax[i].set_title(descriptions[i])


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
    def __init__(self, generator, discriminator, y_sampler,
                 fig_name='conditional_gan.png'):
        self.generator = generator
        self.discriminator = discriminator
        self.fig = plt.figure()
        self.y_sampler = y_sampler
        self.cd = ConditionDescriptor(y_sampler.conditions)
        self.l_ax = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        img_ax0 = plt.subplot2grid((2, 4), (0, 2))
        img_ax1 = plt.subplot2grid((2, 4), (0, 3))
        img_ax2 = plt.subplot2grid((2, 4), (1, 2))
        img_ax3 = plt.subplot2grid((2, 4), (1, 3))
        self.img_ax = [img_ax0, img_ax1, img_ax2, img_ax3]
        #plt.tight_layout()
        self.d_losses = []
        self.g_losses = []
        self.Z = torch.Tensor(4, self.generator.z_size).cuda()
        self.fig_name = fig_name

    def __call__(self, g_loss, d_loss):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        g_handle, = self.l_ax.plot(self.g_losses, 'r', label="Generator loss")
        d_handle, = self.l_ax.plot(self.d_losses, 'b', label="Discriminator loss")
        self.l_ax.legend(handles=[g_handle, d_handle])

        self.Z.uniform_(-1., 1.)
        Y = self.y_sampler.sample_batch(4).cuda()
        G_sample = self.generator(self.Z, Y)
        descriptions = [self.cd.describe(y) for y in Y]
        plot_batch(G_sample.data.cpu().numpy(), normalize=True, ax=self.img_ax, descriptions=descriptions)
        plt.savefig(self.fig_name)
        display.display(plt.gcf())
        display.clear_output(wait=True)


class ConditionDescriptor(object):
    """
    Utility class that can create condition vector y from description
    or generate text description for y
    """
    def __init__(self, conditions):
        self.conditions = conditions
        self.cond_dict = dict(zip(conditions, range(len(conditions))))
        #print(self.cond_dict)

    def create_y(self, **kwargs):
        """
        Elements of y corresponding to keys set to True would be set to 1.0,
        all remaining elements would be set to -1.0
        :param kwargs: 'Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat' or other bool conditions from list_attr_celeba
        :return np.array: generated y

        >>> cd = ConditionDescriptor(['Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat'])
        >>> cd.create_y(Male=True, Young=True)
        [1., -1., 1., -1., -1.]
        """
        y = np.zeros((len(self.conditions),), dtype=np.float32)
        for key, value in kwargs.items():
            if key not in self.cond_dict:
                raise KeyError('No condition named ' + key)
            #print(self.cond_dict[key])
            y[self.cond_dict[key]] = value
        y = y*2 - 1
        return y

    def describe(self, y):
        """
        Generate text description for conditions y
        :param np.array y: condition vector
        :return: text description

        >>> cd = ConditionDescriptor(['Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat'])
        >>> cd.describe(np.array([1., 1., -1., -1., -1.]))
        Smiling old man

        """
        description = ""
        processed_conditions = set()
        if 'Smiling' in self.cond_dict and y[self.cond_dict['Smiling']] > 0:
            description += "Smiling "
        processed_conditions.add('Smiling')
        if 'Young' in self.cond_dict and y[self.cond_dict['Young']] < 0:
            description += "old "
        processed_conditions.add('Young')
        if 'Male' in self.cond_dict:
            if y[self.cond_dict['Male']] > 0:
                description += "man "
            else:
                description += "woman "
        processed_conditions.add('Male')
        description += '\n'
        if 'Eyeglasses' in self.cond_dict and y[self.cond_dict['Eyeglasses']] > 0:
            description += 'with eyeglasses '
        processed_conditions.add('Eyeglasses')

        for i in range(len(y)):
            if self.conditions[i] not in processed_conditions and y[i] > 0:
                description += self.conditions[i].lower().replace('_', ' ') + ' '

        return description
