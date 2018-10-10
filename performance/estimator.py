
from fid.inception import InceptionV3
from fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
import torch
import numpy as np


class Estimator(object):
    def __init__(self):
        pass


class FIDEstimator(Estimator):
    def __init__(self, noise_sampler, device='cuda:1', limit=100):
        super().__init__()
        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.device = torch.device(device)
        self.model = InceptionV3([block_idx]).to(self.device).eval()
        self.noise_sampler = noise_sampler
        self.limit = limit

    def score(self, generator, loader):
        """
        Evaluate generator performance
        :param generator:
        :param loader:
        :return:
        """
        generator.eval()
        X_real = []
        X_fake = []
        for idx, sample in zip(range(self.limit), loader):
            noise = self.noise_sampler.sample_batch(loader.batch_size)
            G_sample = generator(*noise)
            X_real.append(sample[0].numpy())
            X_fake.append(G_sample.data.cpu().numpy()[:sample[0].shape[0], ...])
        X_real = np.concatenate(X_real)
        X_fake = np.concatenate(X_fake)
        return self.distance(X_real, X_fake)

    def distance(self, X1, X2):
        assert(X1.shape == X2.shape)
        X1 += 1
        X1 /= 2
        X2 += 1
        X2 /= 2
        with torch.cuda.device(self.device.index):
            m1, s1 = calculate_activation_statistics(X1, self.model, batch_size=64, dims=self.dims, cuda=1)
            m2, s2 = calculate_activation_statistics(X2, self.model, batch_size=64, dims=self.dims, cuda=1)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


