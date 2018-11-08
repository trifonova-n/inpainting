from inpainting.dataset import Data, ResizeTransform, NoiseSampler
from gan.gan import Generator5Net, Discriminator5
from gan.trainer import GanTrainer
import torch
print(torch.__version__)
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import pandas as pd
from inpainting.visualize import plot_batch
from inpainting.visualize import GanPlotLossCallback as PlotLossCallback
from inpainting import celeba_config as conf
from inpainting.visualizer import Visualizer
from performance.estimator import FIDEstimator

device = torch.device(conf.DEVICE)


transform = ResizeTransform()
data = Data(conf.DATA_PATH, transform)
train_size = int(0.8 * len(data))
valid_size = len(data) - train_size
train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
train_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
print('Dataset size: ', len(data))
noise_sampler = NoiseSampler(conf.Z_SIZE)

estimator = FIDEstimator(noise_sampler, config=conf)

generator = Generator5Net(conf.Z_SIZE).to(device)
discriminator = Discriminator5().to(device)

visualizer = Visualizer(conf, noise_sampler)
trainer = GanTrainer(generator, discriminator, conf, noise_sampler, visualizer=visualizer, estimator=estimator)

if conf.CONTINUE_TRAINING:
    trainer.load_checkpoint(40)

trainer.train(train_loader, valid_loader, n_epochs=100)
