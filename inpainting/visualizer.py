import visdom
import numpy as np
import torch
from inpainting.visualize import ConditionDescriber


class Visualizer(object):
    def __init__(self, env_name, y_sampler):
        self.vis = visdom.Visdom(use_incoming_socket=False, env=env_name)
        self.env_name = env_name
        assert self.vis.check_connection()
        self.vis.text('Hello, world!')
        self.train_losses_plt = 'train_losses'
        self.valid_losses_plt = 'val_losses'
        self.epoch = 0
        self.y_sampler = y_sampler
        self.cd = ConditionDescriber(y_sampler.conditions)

    def update_losses(self, g_loss, d_loss, type):
        if type == 'validation':
            win = self.valid_losses_plt
        elif type == 'train':
            win = self.train_losses_plt
        else:
            print('Unknown losses type: ', type)
            return
        self.epoch += 1
        Y = np.array([[g_loss, d_loss]])
        X = np.array([self.epoch])
        self.vis.line(Y=Y, X=X, win=win, update='append', opts=dict(legend=['generator', 'discriminator']))

    def plot_batch(self, batch, descriptions):
        caption = ', '.join(descriptions)
        self.vis.images(batch, opts=dict(caption=caption))

    def show_generator_results(self, generator):
        Z = torch.Tensor(4, generator.z_size).uniform_(-1., 1.).cuda()
        Y = self.y_sampler.sample_batch(4).cuda()
        G_sample = generator(Z, Y)
        descriptions = [self.cd.describe(y) for y in Y]
        self.plot_batch(G_sample, descriptions)

