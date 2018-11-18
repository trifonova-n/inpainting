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
from argparse import ArgumentParser
import pandas as pd
import pickle

random_state_file = 'random_state.bin'
results_file = 'results_df.scv'

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
    for key in sorted(template.keys()):
        val = template[key]
        if isinstance(val, list):
            idx = randomState.choice(len(val))
            params[key] = val[idx]
        else:
            params[key] = val
    return params


def create_results_row(version, config, score, train_time, valid_time):
    params = {'version': version,
              'score': score,
              'train_time': train_time,
              'valid_time': valid_time,
              **config.training_params,
              **{'g_' + k: v for (k, v) in config.generator_params.items()},
              **{'d_' + k: v for (k, v) in config.discriminator_params.items()}}
    return pd.Series(params)


def save_random_state(path, randomState):
    with (path / random_state_file).open('wb') as f:
        pickle.dump(randomState.get_state(), f)


def load_random_state(path):
    with (path / random_state_file).open('rb') as f:
        randomState = np.random.RandomState()
        randomState.set_state(pickle.load(f))
        return randomState


def train_with_params(config, train_data, valid_data, seed, n_epochs):
    try:
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

        trainer.train(train_loader, n_epochs=n_epochs - 5, save_interval=5)
        # validate only last 5 epochs
        trainer.train(train_loader, valid_loader, n_epochs=5, save_interval=5)
        score = np.mean(trainer.scores[-5:])
        visualizer.log_text('score: %f' % score)
        visualizer.log_text('Training time: %f' % trainer.training_time)
        visualizer.log_text('Validation time: %f' % trainer.validation_time)
        visualizer.log_text(json.dumps(config, indent=2))
    except RuntimeError as e:
        print(e)
        torch.cuda.empty_cache()
        return np.nan, 0, 0

    return score, trainer.training_time, trainer.validation_time


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--model_dir', default='tune_hyperparams')
    parser.add_argument('--env', default='gan_hyperparams')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--n_models', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=3)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    seed = args.seed
    config = Config()
    generator_template = GeneratorParamsTemplate()
    discriminator_template = DiscriminatorParamsTemplate()
    training_template = TrainingParamsTemplate()
    config.DEVICE = args.device
    config.ESTIMATOR_DEVICE = args.device

    transform = ResizeTransform()
    data = Data(config.DATA_PATH, transform)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    start_version = 0
    best_version = 0
    best_score = 10000.0
    randomState = np.random.RandomState(args.seed)
    results_df = pd.DataFrame()
    if args.continue_training:
        results_df.from_csv(str(model_dir / results_file))
        randomState = load_random_state(model_dir)
        start_version = results_df.iloc[-1, :]['version']

    for version in range(start_version, args.n_models):
        config.MODEL_PATH = str(model_dir / ('model_' + str(version)))
        config.ENV_NAME = args.env + '_' + str(version)
        state = randomState.get_state()
        generator_params = generate_params(generator_template, randomState)
        discriminator_params = generate_params(discriminator_template, randomState)
        training_params = generate_params(training_template, randomState)
        config.generator_params = generator_params
        config.discriminator_params = discriminator_params
        config.training_params = training_params
        config.Z_SIZE = generator_params.z_size

        save_path = Path(config.MODEL_PATH)
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'config.json').open('w') as f:
            json.dump(config, f, indent=2)
        print('Train version %d' % version)
        score, train_time, valid_time = train_with_params(config, train_data, valid_data, args.seed, args.n_epochs)
        print('Version %d score: %f' % (version, score))
        if score < best_score:
            best_score = score
            best_version = version
            print('Current best score: ', score)

        row = create_results_row(version, config, score, train_time, valid_time)
        results_df = results_df.append(row, ignore_index=True)

        results_df.to_csv(str(model_dir / results_file))
        save_random_state(model_dir, randomState)
