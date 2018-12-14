import pytest
import torch
from torch.utils.data import DataLoader, random_split
from gan.trainer import GanTrainer
from gan.gan import Generator5Net, Discriminator5
from inpainting.dataset import Data, ResizeTransform, NoiseSampler


@pytest.fixture
def empty_models(conf):
    device = torch.device(conf.DEVICE)
    generator = Generator5Net(conf.Z_SIZE).to(device)
    discriminator = Discriminator5().to(device)
    return generator, discriminator


@pytest.fixture
def empty_trainer(empty_models, conf):
    generator, discriminator = empty_models
    noise_sampler = NoiseSampler(conf.Z_SIZE)
    trainer = GanTrainer(generator, discriminator, conf, noise_sampler)
    return trainer


def test_optimizer_loading(trainer, empty_trainer, dataloaders):
    train_loader, valid_loader = dataloaders
    trainer.save_checkpoint()
    empty_trainer.load_checkpoint(2)
    empty_trainer.train(train_loader, n_epochs=4)

    for p, pe in zip(trainer.generator.parameters(), empty_trainer.generator.parameters()):
        assert torch.allclose(p.data, pe.data)
    for p, pe in zip(trainer.discriminator.parameters(), empty_trainer.discriminator.parameters()):
        assert torch.allclose(p.data, pe.data)
    optimizer = trainer.d_optimizer
    empty_optimizer = empty_trainer.d_optimizer
    for g, ge in zip(optimizer.param_groups, empty_optimizer.param_groups):
        for k in g:
            if k != 'params':
                assert ge[k] == g[k]
        for p, pe in zip(g['params'], ge['params']):
            print(1)
            assert torch.allclose(p.data, pe.data)
            assert torch.allclose(optimizer.state[p]['exp_avg'], empty_optimizer.state[pe]['exp_avg'])
            assert torch.allclose(optimizer.state[p]['exp_avg_sq'], empty_optimizer.state[pe]['exp_avg_sq'])
            assert optimizer.state[p]['step'] == empty_optimizer.state[pe]['step']

