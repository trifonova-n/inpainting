from skimage.io import imread
from skimage.transform import resize
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class ResizeTransform(object):
    def __init__(self, output_shape=(64, 64), sigma=0.05):
        self.output_shape = output_shape
        self.sigma = sigma

    def __call__(self, img):
        img = resize(img, self.output_shape, mode='constant')
        #noise = np.random.normal(0, self.sigma, img.shape)
        return img# + noise


class Data(Dataset):
    def __init__(self, path, z_size, transform=None):
        self.path = Path(path)
        self.z_size = z_size
        self.transform = transform
        self.list_files = sorted(self.path.glob('*.jpg'))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img = imread(str(self.list_files[idx]))
        img = img.astype(np.float32) / 255.
        img = self.transform(img)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        z = np.random.uniform(-1., 1.0, size=(self.z_size,)).astype(np.float32)
        return img, z
