# %%
import h5py

import matplotlib.pyplot as plt
import numpy as np
from hbbgbb import data
import settings
import glob
from tqdm import tqdm
from hbbgbb import eng

DATADIR = 'explore_output'
IMG_SIZE = eng.IMG_SIZE
# %%
def plt_img(data):
    shape = (5, 4)
    start = 0
    fig, ax = plt.subplots(shape[0], shape[1], figsize = (IMG_SIZE, IMG_SIZE), sharey = True)
    fig.patch.set_facecolor('grey')
    for elem in range(1, (shape[0] * shape[1]) + 1):
        plt.subplot(shape[0], shape[1], elem)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data[elem + start])

# %%
def avg_img_perlabel(img_data, labels, name = 'avg_img_perlabel'):
    fig, ax = plt.subplots(1, 3, sharex = True)

    fig.patch.set_facecolor('white')
    for i in range(3):
        elems = eval(f"labels.label{i}")
        cum_img = np.sum(img_data[elems], axis = 0)
        cum_img = np.divide(cum_img, len(elems))
        plt.subplot(1, 3, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cum_img)
        plt.title(f"label {i}")
        plt.colorbar()
    if len(name) > 0:
        plt.savefig(f'{DATADIR}/{name}.pdf')
    else:
        plt.show()
# %%
def test_dist(img_data, labels, func = eng.Feature_Eng.current):
    fig, ax = plt.subplots(3, 2, figsize = (12, 12))
    fig.patch.set_facecolor('white')
    for i in range(3):
        elems = img_data[eval(f"labels.label{i}")]
        elems = elems[elems != 0]
        ax1 = ax[i, 0]
        n, b, p = ax1.hist(elems)
        ax1.set_title(f"label {i} Original")
    img_data = func(img_data)
    for i in range(3):
        elems = img_data[eval(f"labels.label{i}")]
        elems = elems[elems != 0]
        
        ax1 = ax[i, 1]
        n, b, p = ax1.hist(elems)
        ax1.set_title(f"label {i} Engineered")
    plt.tight_layout()
    plt.suptitle(f"{func.__name__} Feature Engineering")
    plt.savefig(f"{DATADIR}/img_dist.pdf")