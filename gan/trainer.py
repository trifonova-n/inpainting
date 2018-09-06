from gan.checkpoint import get_last_checkpoint, load_checkpoint, save_checkpoint
import torch


class BaseGanTrainer(object):
    def __init__(self, generator, discriminator, config, lr=0.0002, visualiser=None):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr, betas=(0.5, 0.999))
        self.visualizer = visualiser
        self.current_epoch = 0

    def train_epoch(self, loader):
        raise NotImplementedError()

    def train(self, loader, n_epochs):
        last_epoch = self.current_epoch
        for self.current_epoch in range(last_epoch + 1, n_epochs):
            self.train_epoch(loader)
            save_checkpoint(self, )

    def load_last_checkpoint(self):
        model_path = self.config.MODEL_PATH
        last_epoch = get_last_checkpoint(model_path)
        if last_epoch >= 0:
            load_checkpoint(model_path, last_epoch, self)


class GanTrainer(BaseGanTrainer):
    def __init__(self, generator, discriminator, config, lr=0.0002, visualiser=None):
        super().__init__(generator, discriminator, config, lr, visualiser)

    def train_epoch(self, loader):
        pass


class ConditionalGanTrainer(BaseGanTrainer):
    def __init__(self, generator, discriminator, config, y_sampler, lr=0.0002, visualiser=None):
        super().__init__(generator, discriminator, config, lr, visualiser)
        self.y_sampler = y_sampler

    def train_epoch(self, loader):
        pass
