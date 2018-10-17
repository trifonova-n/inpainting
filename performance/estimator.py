
from fid.inception import InceptionV3
from fid.fid_score import calculate_activation_statistics, calculate_frechet_distance, get_activations
import torch
import numpy as np


class Estimator(object):
    def __init__(self):
        pass


class FIDEstimator(Estimator):
    def __init__(self, noise_sampler, config, limit=100):
        super().__init__()
        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.generator_device = torch.device(config.DEVICE)
        if hasattr(config, 'ESTIMATOR_DEVICE'):
            self.device = torch.device(config.ESTIMATOR_DEVICE)
        else:
            self.device = self.generator_device
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
        act_real = []
        act_fake = []
        for idx, sample in zip(range(self.limit), loader):
            noise = self.noise_sampler.sample_batch(sample[0].shape[0])
            noise = [c.to(self.generator_device) for c in noise]
            G_sample = (generator(*noise) + 1)/2
            G_sample = G_sample.to(self.device)
            with torch.cuda.device(self.device.index):
                act_real_batch = get_activations(((sample[0] + 1)/2).cuda(), self.model, batch_size=64, dims=self.dims, cuda=1)
                act_fake_batch = get_activations(G_sample, self.model, batch_size=64, dims=self.dims, cuda=1)
                act_real.append(act_real_batch)
                act_fake.append(act_fake_batch)

        act_real = np.concatenate(act_real)
        act_fake = np.concatenate(act_fake)
        m1 = np.mean(act_real, axis=0)
        s1 = np.cov(act_real, rowvar=False)
        m2 = np.mean(act_fake, axis=0)
        s2 = np.cov(act_fake, rowvar=False)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

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


