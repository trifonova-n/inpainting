import visdom
import numpy as np


class Visualizer(object):
    def __init__(self, env_name):
        self.vis = visdom.Visdom(use_incoming_socket=False)
        assert self.vis.check_connection()
        self.vis.text('Hello, world!')
        self.train_losses_plt = 'train_losses'
        self.valid_losses_plt = 'val_losses'
        self.epoch = 0

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
