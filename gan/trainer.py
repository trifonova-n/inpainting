from gan.checkpoint import get_last_checkpoint, load_checkpoint, save_checkpoint
from gan.losses import GeneratorLoss, DiscriminatorLoss
import torch
import itertools


class BaseGanTrainer(object):
    def __init__(self, generator, discriminator, config, lr=0.0002, visualizer=None):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr, betas=(0.5, 0.999))
        self.visualizer = visualizer
        self.current_epoch = 0

    def train_epoch(self, loader):
        raise NotImplementedError()

    def train(self, loader, n_epochs):
        last_epoch = self.current_epoch
        for self.current_epoch in range(last_epoch + 1, n_epochs):
            self.train_epoch(loader)
            save_checkpoint(self, self.config.MODEL_PATH)

    def load_last_checkpoint(self):
        model_path = self.config.MODEL_PATH
        last_epoch = get_last_checkpoint(model_path)
        if last_epoch >= 0:
            load_checkpoint(model_path, last_epoch, self)


class GanTrainer(BaseGanTrainer):
    def __init__(self, generator, discriminator, config, lr=0.0002, visualizer=None):
        super().__init__(generator, discriminator, config, lr, visualizer)

    def train_epoch(self, loader, k=2):
        self.generator.train()
        self.discriminator.train()
        G_train_loss = 0.0
        D_train_loss = 0.0
        n_d_steps = 0
        n_g_steps = 0
        k_it = 0
        generator_loss = GeneratorLoss()
        discriminator_loss = DiscriminatorLoss(label_smoothing=0.25)

        for img, Z in loader:
            X = img.cuda()
            Z = Z.cuda()
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()

            G_sample = self.generator(Z)
            D_real, D_logit_real = self.discriminator(X)
            D_fake, D_logit_fake = self.discriminator(G_sample)

            D_loss = discriminator_loss(D_logit_real, D_logit_fake)
            D_train_loss += D_loss.data

            D_loss.backward()
            self.d_optimizer.step()
            n_d_steps += 1

            k_it += 1

            if k_it == k:
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                Z.uniform_(-1., 1.)
                G_sample = self.generator(Z)
                D_fake, D_logit_fake = self.discriminator(G_sample)
                G_loss = generator_loss(D_logit_fake)
                G_train_loss += G_loss.data

                G_loss.backward()
                self.g_optimizer.step()
                k_it = 0
                n_g_steps += 1

        if self.visualizer is not None:
            self.visualizer.update_losses(g_loss=G_train_loss / n_g_steps, d_loss=D_train_loss / n_d_steps, type='train')


class ConditionalGanTrainer(BaseGanTrainer):
    def __init__(self, generator, discriminator, config, y_sampler, lr=0.0002, visualizer=None):
        super().__init__(generator, discriminator, config, lr, visualizer)
        self.y_sampler = y_sampler

    def train_epoch(self, loader):
        self.generator.train()
        self.discriminator.train()
        G_train_loss = 0.0
        D_train_loss = 0.0
        n_d_steps = 0
        n_g_steps = 0
        k_it = 0
        generator_loss = GeneratorLoss()
        discriminator_loss = DiscriminatorLoss(label_smoothing=self.config.label_smoothing)

        for img, Z, Y in itertools.islice(loader, 10):
            X = img.cuda()
            Z = Z.cuda()
            Y = Y.cuda()
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()

            G_sample = self.generator(Z, Y)
            D_real, D_logit_real = self.discriminator(X, Y)
            D_fake, D_logit_fake = self.discriminator(G_sample, Y)

            D_loss = discriminator_loss(D_logit_real, D_logit_fake)
            D_train_loss += D_loss.data

            D_loss.backward()
            self.d_optimizer.step()
            n_d_steps += 1

            k_it += 1

            if k_it == self.config.k:
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                Z = torch.FloatTensor(loader.batch_size, self.generator.z_size).uniform_(-1., 1.).cuda()
                Y = self.y_sampler.sample_batch(loader.batch_size)
                # print(Z.shape, Y.shape)
                Y = Y.cuda()
                G_sample = self.generator(Z, Y)
                D_fake, D_logit_fake = self.discriminator(G_sample, Y)
                G_loss = generator_loss(D_logit_fake)
                G_train_loss += G_loss.data

                G_loss.backward()
                self.g_optimizer.step()
                k_it = 0
                n_g_steps += 1

        if self.visualizer is not None:
            self.visualizer.update_losses(g_loss=G_train_loss / n_g_steps, d_loss=D_train_loss / n_d_steps, type='train')
            self.visualizer.show_generator_results(generator=self.generator)

