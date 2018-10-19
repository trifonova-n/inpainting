from skimage.io import imread
from skimage.transform import resize
from skimage.util import crop
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
import pandas as pd


class ResizeTransform(object):
    def __init__(self, output_shape=(64, 64), sigma=0.05):
        self.output_shape = output_shape
        self.sigma = sigma

    def __call__(self, img):
        width = img.shape[1]
        height = img.shape[0]
        box_side = 140
        horisontal = (width - box_side) // 2
        vertical = (height - box_side) // 2
        img = img[vertical + 10:height - vertical + 10, horisontal:width - horisontal]
        img = resize(img, self.output_shape, mode='constant')
        return img


class Data(Dataset):
    def __init__(self, path, transform=None, return_attr=False,
                 conditions=('Male', 'Smiling', 'Young', 'Eyeglasses', 'Wearing_Hat')):
        """
        Data class returns tuple with one element (img,) if return_attr=False
        or with two elements (img, y) if return_attr=True
        img is array with shape [3, 64, 64] and elements in range [-1, 1]
        y is array with shape [len(conditions)] and elements in range [-1, 1]
        :param path: data path
        :param transform: transformation function
        :param return_attr: False for unconditional gan, True for conditional
        :param conditions: list of names of conditions in conditional vector
        """
        self.path = Path(path)
        self.return_attr = return_attr
        self.transform = transform
        self.list_files = sorted(self.path.glob('*.jpg'))
        self.hdf5_file = self.path.parent / (str(self.path.name) + '.hdf5')
        if not self.hdf5_file.exists():
            self.convert_to_h5(path, self.hdf5_file)
        self.load_images()
        # df_attr is pandas Dataframe containing conditional vectors for each image
        self.df_attr = pd.read_csv(str(self.path.parent/'list_attr_celeba.txt'), sep='\s+', header=1)
        self.conditions = list(conditions)
        self.df_attr = self.df_attr[list(conditions)]
        self.y_size = len(self.df_attr.columns)

    def __len__(self):
        return len(self.images)

    def find_image(self, y):
        # find index of random image with conditional vector y
        idx = self.df_attr.index.get_loc(self.df_attr[(self.df_attr == y).all(axis=1)].sample(1).index[0])
        return idx

    def load_image(self, idx):
        img = imread(str(self.list_files[idx]))
        img = img.astype(np.float32) / 255.
        img = self.transform(img)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        # images in range [-1, 1]
        return img * 2 - 1.0

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
        idx = int(idx)
        img = self.images[idx]
        if self.return_attr:
            y = self.df_attr.iloc[idx].values.astype(np.float32)
            return img, y
        else:
            # tuple with 1 element
            return (img,)


class NoiseSampler(object):
    """
    Generates input noise z for generator from uniform distribution [-1., 1.]
    """
    def __init__(self, z_size, seed=1):
        self.z_size = z_size
        self.seed = seed
        self.randomState = np.random.RandomState(self.seed)

    def manual_seed(self, seed):
        self.randomState = np.random.RandomState(self.seed)

    def sample(self):
        z = self.randomState.uniform(-1., 1.0, size=(self.z_size,)).astype(np.float32)
        # tuple with 1 element
        # we need tuple here to have interface consistent with ConditionSampler
        return (torch.from_numpy(z),)

    def sample_batch(self, batch_size):
        z = self.randomState.uniform(-1., 1.0, size=(batch_size, self.z_size)).astype(np.float32)
        # tuple with 1 element
        # we need tuple here to have interface consistent with ConditionSampler
        return (torch.from_numpy(z),)


class ConditionSampler(NoiseSampler):
    """
    Generates input noise z for generator from uniform distribution [-1., 1.]
    and condition y from training data distribution
    """
    def __init__(self, data, z_size):
        super().__init__(z_size)
        self.df_attr = data.df_attr
        self.conditions = data.conditions

    def sample(self):
        z = NoiseSampler.sample(self)[0]
        return z, torch.from_numpy(self.df_attr.sample(1, random_state=self.randomState).iloc[0].values.astype(np.float32))

    def sample_batch(self, batch_size):
        z = NoiseSampler.sample_batch(self, batch_size)[0]
        y = self.df_attr.sample(batch_size, random_state=self.randomState).values.astype(np.float32)
        return z, torch.tensor(y)
