import pytest
import torch
from torch.utils.data import DataLoader, random_split
from gan.trainer import GanTrainer
from gan.gan import Generator5Net, Discriminator5
from inpainting.dataset import Data, ResizeTransform, NoiseSampler


@pytest.fixture
def conf():
    class Config(object):
        DATA_PATH = 'data/img_align_celeba'
        BATCH_SIZE = 256
        NUM_WORKERS = 0
        Z_SIZE = 100
        MODEL_PATH = 'test_model/'
        CONTINUE_TRAINING = False
        DEVICE = 'cuda:0'
        ESTIMATOR_DEVICE = 'cuda:0'
        label_smoothing = 0.25
        k = 1  # how many times to update discriminator for 1 generator update
        ENV_NAME = "test_gan"
        NEW_VISDOM_ENV = True
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
    train_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True, drop_last=True)
    return train_loader, valid_loader


@pytest.fixture
def models(conf):
    device = torch.device(conf.DEVICE)
    generator = Generator5Net(conf.Z_SIZE).to(device)
    discriminator = Discriminator5().to(device)
    generator = generator.train()
    discriminator = discriminator.train()
    return generator, discriminator


@pytest.fixture
def trainer(dataloaders, models, conf):
    train_loader, valid_loader = dataloaders
    generator, discriminator = models
    noise_sampler = NoiseSampler(conf.Z_SIZE)
    trainer = GanTrainer(generator, discriminator, conf, noise_sampler, seed=2)
    #trainer.train(train_loader, n_epochs=4)
    return trainer

