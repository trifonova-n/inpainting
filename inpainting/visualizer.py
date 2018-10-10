import visdom
import numpy as np
import torch
from inpainting.visualize import ConditionDescriber
import json


class Visualizer(object):
    vis = visdom.Visdom(use_incoming_socket=False)

    def __init__(self, config, noise_sampler):
        self.env_name = config.ENV_NAME
        assert self.vis.check_connection()
        self.log_win = 'text_log'
        self.train_losses_plt = 'train_losses'
        self.valid_losses_plt = 'val_losses'
        self.gen_res_img = 'gen_res'
        self.noise_sampler = noise_sampler
        if hasattr(config, 'conditions'):
            self.cd = ConditionDescriber(config.conditions)
        # creates new environment version by default
        # set_env can be used to specify usage of existing environment
        self._set_new_env_version(self.env_name)
        self.text = "Text log:"

    def update_losses(self, epoch, g_loss, d_loss, type):
        if type == 'validation':
            win = self.valid_losses_plt
        elif type == 'train':
            win = self.train_losses_plt
        else:
            print('Unknown losses type: ', type)
            return
        Y = np.array([[g_loss, d_loss]])
        X = np.array([epoch])
        self.vis.line(Y=Y, X=X, win=win, env=self.env_name, update='append',
                      opts=dict(title=type + " losses", legend=['generator', 'discriminator']))
        print("Update losses")

    def update_plot(self, x, y, name):
        Y = np.array([y])
        X = np.array([x])
        self.vis.line(Y=Y, X=X, win=name, env=self.env_name, update='append', opts=dict(title=name))

    def plot_batch(self, batch, descriptions):
        caption = ', '.join(descriptions)
        self.vis.images((batch + 1.0)/2.0, opts=dict(title=caption), env=self.env_name, win=self.gen_res_img)

    def show_generator_results(self, generator):
        noise = self.noise_sampler.sample_batch(4)
        G_sample = generator(*noise)
        # if conditional gan
        if len(noise) > 1:
            Y = noise[1]
            descriptions = [self.cd.describe(y) for y in Y]
        else:
            descriptions = ["Generated images"]
        self.plot_batch(G_sample, descriptions)
        print("show_generator_results")

    def log_text(self, msg):
        self.text += "<br>" + msg
        self.vis.text(self.text, win=self.log_win, env=self.env_name)

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
        if envs:
            last_version = max(int(v) for _, v in map(split_name, envs))
        else:
            last_version = 0
        self.env_name = name + '_' + str(last_version + 1)

    def set_env(self, env_name):
        self.env_name = env_name


