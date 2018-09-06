import visdom
import numpy as np
import torch
from inpainting.visualize import ConditionDescriber
import json


class Visualizer(object):
    vis = visdom.Visdom(use_incoming_socket=False)

    def __init__(self, env_name, y_sampler):
        self.env_name = env_name
        assert self.vis.check_connection()
        self.log_win = 'text_log'
        self.train_losses_plt = 'train_losses'
        self.valid_losses_plt = 'val_losses'
        self.epoch = 0
        self.y_sampler = y_sampler
        self.cd = ConditionDescriber(y_sampler.conditions)
        # creates new environment version by default
        # set_env can be used to specify usage of existing environment
        self._set_new_env_version(env_name)

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
        self.vis.line(Y=Y, X=X, win=win, env=self.env_name, update='append', opts=dict(legend=['generator', 'discriminator']))

    def plot_batch(self, batch, descriptions):
        caption = ', '.join(descriptions)
        self.vis.images(batch, opts=dict(caption=caption))

    def show_generator_results(self, generator):
        Z = torch.Tensor(4, generator.z_size).uniform_(-1., 1.).cuda()
        Y = self.y_sampler.sample_batch(4).cuda()
        G_sample = generator(Z, Y)
        descriptions = [self.cd.describe(y) for y in Y]
        self.plot_batch(G_sample, descriptions)

    def log_text(self, msg):
        self.vis.text(msg, win=self.log_win, env=self.env_name, append=True)

    def save(self):
        self.vis.save([self.env_name])

    def _set_new_env_version(self, env_name):
        def split_name(name):
            fields = name.split('_')
            if fields[-1].isdigit():
                return '_'.join(fields[:-1]), int(fields[-1])
            else:
                return name, 0

        name, version = split_name(env_name)
        envs = [s for s in self.vis.get_env_list() if name in s]
        last_version = max(int(v) for _, v in map(split_name, envs))
        self.env_name = name + '_' + str(last_version + 1)

    def set_env(self, env_name):
        self.env_name = env_name




