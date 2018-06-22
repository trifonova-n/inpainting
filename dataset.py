from skimage.io import imread
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch


class Data(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        self.transform = transform
        self.list_files = sorted(self.path.glob('*.jpg'))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img = imread(str(self.list_files[idx]))
        img = img.astype(np.float32) / 255.
        return img

