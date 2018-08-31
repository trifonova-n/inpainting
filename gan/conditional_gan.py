import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from pathlib import Path
from gan.losses import GeneratorLoss, DiscriminatorLoss
from gan.checkpoint import load_checkpoint, save_checkpoint, get_last_checkpoint


class CGeneratorNet(torch.nn.Module):
    def __init__(self, z_size, y_size):
        super(CGeneratorNet, self).__init__()
        self.z_size = z_size
        self.y_size = y_size
        self.layer0 = nn.Linear(z_size + y_size, 1024*4*4)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.noise = torch.Tensor(64, 64).cuda()
        self.eps = 0.01

    def forward(self, z, y):
        input = torch.cat((z, y), 1)
        out = self.layer0(input).view(-1, 1024, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #self.noise.normal_(0, self.eps)

        return out# + self.noise


class CDiscriminator(torch.nn.Module):
    def __init__(self, y_size):
        super(CDiscriminator, self).__init__()
        self.y_size = y_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        self.layer4 = nn.Linear(512*2*2 + self.y_size, 512)
        self.layer5 = nn.Linear(512, 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).view(-1, 512*2*2)
        out = torch.cat((out, y), 1)
        out = self.layer4(out)
        D_logit = self.layer5(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit


class CGenerator5Net(torch.nn.Module):
    def __init__(self, z_size, y_size):
        super(CGenerator5Net, self).__init__()
        self.z_size = z_size
        self.y_size = y_size
        self.layer0 = nn.Linear(z_size + y_size, 1024*4*4)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.noise = torch.Tensor(64, 64).cuda()
        self.eps = 0.01

    def forward(self, z, y):
        input = torch.cat((z, y), 1)
        out = self.layer0(input).view(-1, 1024, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #self.noise.normal_(0, self.eps)

        return out# + self.noise


class CDiscriminator5(torch.nn.Module):
    def __init__(self, y_size):
        super(CDiscriminator5, self).__init__()
        self.y_size = y_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        self.layer4 = nn.Linear(512*2*2 + self.y_size, 512)
        self.layer5 = nn.Linear(512, 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).view(-1, 512*2*2)
        out = torch.cat((out, y), 1)
        out = self.layer4(out)
        D_logit = self.layer5(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit


def train_epoch(generator, discriminator, G_optimizer, D_optimizer, loader, y_sampler, k=2, callback_func=None):
    generator.train()
    discriminator.train()
    G_train_loss = 0.0
    D_train_loss = 0.0
    n_d_steps = 0
    n_g_steps = 0
    k_it = 0
    #G_grads = []
    #D_grads = []
    #y_sampler = Bernoulli(torch.tensor(p).expand(loader.batch_size, -1))
    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss(label_smoothing=0.25)

    for img, Z, Y in loader:
        X = img.cuda()
        Z = Z.cuda()
        Y = Y.cuda()
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

        G_sample = generator(Z, Y)
        D_real, D_logit_real = discriminator(X, Y)
        D_fake, D_logit_fake = discriminator(G_sample, Y)

        D_loss = discriminator_loss(D_logit_real, D_logit_fake)
        D_train_loss += D_loss.data

        D_loss.backward()
        #D_grads.append(np.mean(np.abs(discriminator.layer1[0].weight.grad.cpu().numpy())))
        D_optimizer.step()
        n_d_steps += 1

        k_it += 1

        if k_it == k:
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()
            Z = torch.FloatTensor(loader.batch_size, generator.z_size).uniform_(-1., 1.).cuda()
            Y = y_sampler.sample_batch(loader.batch_size)
            print(Z.shape, Y.shape)
            Y = Y.cuda()
            G_sample = generator(Z, Y)
            #D_real, D_logit_real = discriminator(X)
            D_fake, D_logit_fake = discriminator(G_sample, Y)
            G_loss = generator_loss(D_logit_fake)
            G_train_loss += G_loss.data

            G_loss.backward()
            #G_grads.append(np.mean(np.abs(generator.layer0.weight.grad.cpu().numpy())))
            #print(np.mean(np.abs(generator.layer0.weight.grad.cpu().numpy())))
            G_optimizer.step()
            k_it = 0
            n_g_steps += 1

        #if n_g_steps >= 1000:
        #    break

    if callback_func is not None:
        #pass
        callback_func(g_loss=G_train_loss / n_g_steps, d_loss=D_train_loss / n_d_steps)


def train(generator, discriminator, loader, y_sampler, n_epochs=100, k=2, callback_func=None, model_path='model', continue_training=False):
    model_path = Path(model_path)
    if continue_training:
        last_epoch = get_last_checkpoint(model_path)
    else:
        last_epoch = -1
    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                                   lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                   lr=0.0002, betas=(0.5, 0.999))

    if last_epoch >= 0:
        load_checkpoint(model_path, last_epoch, generator, discriminator, G_optimizer, D_optimizer)

    for i in range(last_epoch + 1, n_epochs):
        train_epoch(generator, discriminator, G_optimizer, D_optimizer, loader,
                    y_sampler=y_sampler, k=k, callback_func=callback_func)
        save_checkpoint(generator, discriminator,
                        g_optimizer=G_optimizer, d_optimizer=D_optimizer, epoch=i, save_path=model_path)