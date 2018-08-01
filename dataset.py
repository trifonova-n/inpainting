from skimage.io import imread
from skimage.transform import resize
from skimage.util import crop
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import h5py
import pandas as pd


class ResizeTransform(object):
    def __init__(self, output_shape=(64, 64), sigma=0.05, path='data'):
        self.output_shape = output_shape
        self.sigma = sigma

    def __call__(self, img):
        #noise = np.random.normal(0, self.sigma, img.shape)
        width = img.shape[1]
        height = img.shape[0]
        box_side = 140
        horisontal = (width - box_side) // 2
        vertical = (height - box_side) // 2
        img = img[vertical + 10:height - vertical + 10, horisontal:width - horisontal]
        #print(img.shape)
        img = resize(img, self.output_shape, mode='constant')
        return img# + noise


class Data(Dataset):
    def __init__(self, path, z_size, transform=None):
        self.path = Path(path)
        self.z_size = z_size
        self.transform = transform
        self.list_files = sorted(self.path.glob('*.jpg'))
        self.hdf5_file = self.path.parent / (str(self.path.name) + '.hdf5')
        if not self.hdf5_file.exists():
            self.convert_to_h5(path, self.hdf5_file)
        #self.images = np.empty((len(self.list_files), 3, self.transform.output_shape[0], self.transform.output_shape[1]),
        #                       dtype=np.float32)
        self.load_images()
        self.df_attr = pd.read_csv(str(self.path.parent/'list_attr_celeba.txt'), sep='\s+', header=1)
        self.df_attr = self.df_attr[['Male', 'Smiling', 'Young', 'Mustache', 'Bald', 'Eyeglasses', 'Wearing_Hat']]

    def __len__(self):
        return len(self.list_files)

    def load_image(self, idx):
        img = imread(str(self.list_files[idx]))
        img = img.astype(np.float32) / 255.
        img = self.transform(img)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        return img

    def load_images(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            dset = f['images']
            self.images = dset[:]

    def convert_to_h5(self, path, out_path):
        path = Path(path)
        list_files = sorted(path.glob('*.jpg'))
        with h5py.File(out_path, "w") as f:
            dset = f.create_dataset("images", (len(list_files), 3, 64, 64), dtype='f4')
            for i in range(len(self.list_files)):
                img = self.load_image(i)
                dset[i] = img




    def __getitem__(self, idx):
        img = self.images[idx]
        z = np.random.uniform(-1., 1.0, size=(self.z_size,)).astype(np.float32)
        return img, z
