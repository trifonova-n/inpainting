from inpainting.dataset import Data, ResizeTransform, NoiseSampler
from gan.gan import Generator5Net, Discriminator5
from gan.trainer import GanTrainer
import torch
print(torch.__version__)
from torch.utils.data import DataLoader, random_split

from inpainting.visualizer import Visualizer
from performance.estimator import FIDEstimator
from argparse import ArgumentParser
import importlib

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', default='model')
    parser.add_argument('--conf', default='inpainting.celeba_config')
    args = parser.parse_args()
    conf = importlib.import_module(args.conf)
    # torch.cuda.set_device(conf.CUDA_DEVICE)
    device = torch.device(conf.DEVICE)
    if hasattr(conf, 'ESTIMATOR_DEVICE'):
        estimator_device = torch.device(conf.ESTIMATOR_DEVICE)
    else:
        estimator_device = device

    transform = ResizeTransform()
    data = Data(conf.DATA_PATH, transform)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS, shuffle=True)
    print('Dataset size: ', len(data))
    noise_sampler = NoiseSampler(conf.Z_SIZE)

    generator = Generator5Net(conf.Z_SIZE).to(device)
    discriminator = Discriminator5().to(device)
    visualizer = Visualizer(conf, noise_sampler)
    estimator = FIDEstimator(noise_sampler, device=estimator_device)
    trainer = GanTrainer(generator, discriminator, conf, noise_sampler, visualizer=visualizer, estimator=estimator)

    last_epoch = GanTrainer.get_last_checkpoint(conf.MODEL_PATH)
    for epoch in range(1, last_epoch + 1):
        trainer.load_checkpoint(epoch)
        score = estimator.score(generator, loader=valid_loader)
        visualizer.update_plot(epoch, score, 'FID')
        print("Epoch: ", epoch, ": ", score)
