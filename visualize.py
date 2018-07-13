import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def plot_batch(img_batch, transpose_channels=True):
    if transpose_channels:
        img_batch = np.transpose(img_batch, (0, 2, 3, 1))
    n = img_batch.shape[0]
    ncols = int(sqrt(n))
    nrows = n // ncols
    if nrows * ncols < n:
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols)
    ax = axes.ravel()
    for i in range(img_batch.shape[0]):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].imshow(img_batch[i])
