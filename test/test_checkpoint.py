import pytest
import torch
from torch.utils.data import DataLoader, random_split
from gan.trainer import GanTrainer
from gan.gan import Generator5Net, Discriminator5
from inpainting.dataset import Data, ResizeTransform, NoiseSampler


@pytest.fixture
def conf():
    class Config(object):
        DATA_PATH = 'test_data/img_align_celeba'
        BATCH_SIZE = 4
        NUM_WORKERS = 0
        Z_SIZE = 100
        MODEL_PATH = 'test_model/'
        CONTINUE_TRAINING = False
        DEVICE = 'cuda:0'
        ESTIMATOR_DEVICE = 'cuda:0'
        label_smoothing = 0.25
        k = 1  # how many times to update discriminator for 1 generator update
        ENV_NAME = "test_gan"
    return Config()


@pytest.fixture
def data(conf):
    transform = ResizeTransform()
    data = Data(conf.DATA_PATH, transform)
    return data


@pytest.fixture
def dataloaders(data, conf):
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
    return train_loader, valid_loader


@pytest.fixture
def models(conf):
    device = torch.device(conf.DEVICE)
    generator = Generator5Net(conf.Z_SIZE).to(device)
    discriminator = Discriminator5().to(device)
    return generator, discriminator


@pytest.fixture
def trainer(dataloaders, models, conf):
    train_loader, valid_loader = dataloaders
    generator, discriminator = models
    noise_sampler = NoiseSampler(conf.Z_SIZE)
    trainer = GanTrainer(generator, discriminator, conf, noise_sampler)
    trainer.train(train_loader, n_epochs=3)
    return trainer


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
    empty_trainer.train(train_loader, n_epochs=3)

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

