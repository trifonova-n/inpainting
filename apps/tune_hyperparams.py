from inpainting.dataset import Data, ResizeTransform, NoiseSampler
from gan.gan import GeneratorNet, DiscriminatorNet, Generator5Net, Discriminator5
from gan.trainer import GanTrainer
import torch
from torch.utils.data import DataLoader, random_split
from inpainting.celeba_config import Config
from inpainting.visualizer import Visualizer
from performance.estimator import FIDEstimator
from gan.hyperparameters import GeneratorParams, DiscriminatorParams, Params
import numpy as np
import json
from pathlib import Path


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


class TrainingParamsTemplate(Params):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = [0.0002, 0.0001, 0.001, 0.0005]
        self.generator_weight_decay = [5e-5, 1e-5, 5e-6]
        self.discriminator_weight_decay = [5e-5, 1e-5, 5e-6]



def generate_params(template, randomState):
    params = Params()
    for key, val in template.items():
        if isinstance(val, list):
            params[key] = randomState.choice(val)
        else:
            params[key] = val
    return params


def train_with_params(config, train_data, valid_data, seed):
    device = torch.device(config.DEVICE)

    generator = GeneratorNet(params=config.generator_params).to(device)
    discriminator = DiscriminatorNet(params=config.discriminator_params).to(device)

    noise_sampler = NoiseSampler(config.Z_SIZE)
    estimator = FIDEstimator(noise_sampler, config=config, limit=10000)

    visualizer = Visualizer(config, noise_sampler)
    visualizer.env_name = config.ENV_NAME
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
    trainer = GanTrainer(generator=generator,
                         discriminator=discriminator,
                         config=config,
                         noise_sampler=noise_sampler,
                         visualizer=visualizer,
                         estimator=estimator,
                         seed=seed)

    trainer.train(train_loader, n_epochs=25, save_interval=5)
    # validate only last 5 epochs
    trainer.train(train_loader, valid_loader, n_epochs=5, save_interval=5)
    score = np.mean(trainer.scores[-5:])
    visualizer.log_text('score: %f' % score)
    visualizer.log_text('Training time: %f' % trainer.training_time)
    visualizer.log_text('Validation time: %f' % trainer.validation_time)
    visualizer.log_text(json.dumps(config))
    return score


if __name__ == '__main__':
    seed = 1
    config = Config()
    generator_template = GeneratorParamsTemplate()
    discriminator_template = DiscriminatorParamsTemplate()
    training_template = TrainingParamsTemplate()
    config.DEVICE = 'cuda:1'
    config.ESTIMATOR_DEVICE = 'cuda:1'
    transform = ResizeTransform()
    data = Data(config.DATA_PATH, transform)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    best_version = 0
    best_score = 10000.0
    randomState = np.random.RandomState(seed)
    for version in range(20):
        config.MODEL_PATH = 'model_' + str(version) + '/'
        config.ENV_NAME = 'gan_hyperparams_' + str(version)
        state = randomState.get_state()
        generator_params = generate_params(generator_template, randomState)
        discriminator_params = generate_params(discriminator_template, randomState)
        training_params = generate_params(training_template, randomState)
        config.generator_params = generator_params
        config.discriminator_params = discriminator_params
        config.training_params = training_params
        config.Z_SIZE = generator_params.z_size

        save_path = Path(config.MODEL_PATH)
        save_path.mkdir(exist_ok=True)
        with (save_path / 'config.json').open('w') as f:
            json.dump(config, f)
        score = train_with_params(config, train_data, valid_data, seed)
        print('Version %d score: %f', (version, score))
        if score < best_score:
            best_score = score
            best_version = version
            print('Current best score: ', score)
