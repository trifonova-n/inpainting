import torch
import torch.nn as nn
import numpy as np


class GeneratorNet(torch.nn.Module):
    def __init__(self, z_size):
        super(GeneratorNet, self).__init__()
        self.z_size = z_size
        self.layer0 = nn.Linear(z_size, 64*4*4)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())
        self.noise = torch.Tensor(64, 64).cuda()
        self.eps = 0.01

    def forward(self, z):
        out = self.layer0(z).view(-1, 64, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        self.noise.normal_(0, self.eps)

        return out + self.noise


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Linear(64*2*2, 1)
        self.out_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).view(-1, 64*2*2)
        D_logit = self.layer4(out)
        D_prob = self.out_layer(D_logit)
        return D_prob, D_logit


def generator_loss(D_fake):
    #return torch.mean(torch.log(1. - D_fake))
    return -torch.mean(torch.log(D_fake))


def discriminator_loss(D_real, D_fake):
    return -torch.mean(torch.log(D_real) + torch.log(1. - D_fake))


def train_epoch(generator, discriminator, G_optimizer, D_optimizer, loader, k=2, callback_func=None):
    generator.train()
    discriminator.train()
    G_train_loss = 0.0
    D_train_loss = 0.0
    n_d_steps = 0
    n_g_steps = 0
    k_it = 0

    for img, Z in loader:
        X = img.cuda()
        Z = Z.cuda()
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

        G_sample = generator(Z)
        D_real, D_logit_real = discriminator(X)
        D_fake, D_logit_fake = discriminator(G_sample)

        D_loss = discriminator_loss(D_real, D_fake)
        D_train_loss += D_loss.data

        D_loss.backward()
        D_optimizer.step()
        n_d_steps += 1

        k_it += 1

        if k_it == k:
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()
            Z.uniform_(-1., 1.)
            G_sample = generator(Z)
            #D_real, D_logit_real = discriminator(X)
            D_fake, D_logit_fake = discriminator(G_sample)
            G_loss = generator_loss(D_fake)
            G_train_loss += G_loss.data

            G_loss.backward()
            #print(np.mean(np.abs(generator.layer0.weight.grad.cpu().numpy())))
            G_optimizer.step()
            k_it = 0
            n_g_steps += 1

        #if n_g_steps >= 1000:
        #    break

    if callback_func is not None:
        #pass
        callback_func(g_loss=G_train_loss / n_g_steps, d_loss=D_train_loss / n_d_steps)


def train(generator, discriminator, loader, n_epochs=100, k=2, callback_func=None):
    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                                 lr=0.00002, betas=(0.5, 0.999), weight_decay=0.0001)
    D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                 lr=0.00002, betas=(0.5, 0.999), weight_decay=0.0001)

    for i in range(n_epochs):
        train_epoch(generator, discriminator, G_optimizer, D_optimizer, loader, k=k, callback_func=callback_func)

