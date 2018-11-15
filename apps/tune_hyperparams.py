from inpainting.dataset import Data, ResizeTransform, NoiseSampler
from gan.gan import GeneratorNet, DiscriminatorNet
from gan.trainer import GanTrainer
import torch
from torch.utils.data import DataLoader, random_split
from inpainting import celeba_config as conf
from inpainting.visualizer import Visualizer
from performance.estimator import FIDEstimator
from gan.hyperparameters import GeneratorParams, DiscriminatorParams, Params
import numpy as np


class GeneratorParamsTemplate(GeneratorParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_shape = [(1024, 4, 4), (1024, 2, 2), (512, 4, 4), (512, 2, 2)]
        self.channel_scaling_factor = [2, 1.5]
        self.z_size = [100, 200]
        self.bn_start_idx = [0, 1, 2]
        self.bn_end_idx = [-1, -2, -3]


class DiscriminatorParamsTemplate(DiscriminatorParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_channels = [16, 32, 64]
        self.feature_img_size = [2, 4]
        self.channel_scaling_factor = [2, 1.5]
        self.bn_start_idx = [0, 1, 2]
        self.bn_end_idx = [-2, -3]


def generate_params(template):
    params = Params()
    for key, val in template.items():
        if isinstance(val, list):
            params[key] = np.random.choice(val)
        else:
            params[key] = val
    return params


def train_with_params(generator_params, discriminator_params, config, train_data, valid_data):
    config.Z_SIZE = generator_params.z_size
    config.ENV_NAME = 'gan_hyperparams'
    device = torch.device(conf.DEVICE)

    generator = GeneratorNet(params=generator_params).to(device)
    discriminator = DiscriminatorNet(params=discriminator_params).to(device)

    noise_sampler = NoiseSampler(config.Z_SIZE)
    estimator = FIDEstimator(noise_sampler, config=conf, limit=10000)

    visualizer = Visualizer(conf, noise_sampler)
    train_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
    trainer = GanTrainer(generator=generator,
                         discriminator=discriminator,
                         config=config,
                         noise_sampler=noise_sampler,
                         visualizer=visualizer,
                         estimator=estimator,
                         seed=1)

    trainer.train(train_loader, valid_loader, n_epochs=20)
    score = np.mean(trainer.scores[-5:])
    return score
