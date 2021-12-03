# %% Imports 
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
from file_support import *
from model_support import *
import math
import time
#from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
# %%
#Directory and label file selection
BGROUND_Label = {0:[1]}
M_SIG_Label = {0:[0]}
I_SIG_Label = {0:[2]}
MODELTYPE = 'cnn'
# Retrieve Datasets
dataset_arr = []
label_arr = []
for ev_type in [BGROUND_Label, M_SIG_Label, I_SIG_Label]:
    if ev_type == BGROUND_Label:
        ev_dir = BGROUND
    elif ev_type == M_SIG_Label:
        ev_dir = M_SIG
    else:
        ev_dir = I_SIG
    for event in ev_type.keys():
        for label in ev_type[event]:
            curr_arr = np.load(f"{DATA_DIR}/{ev_dir[event]}/label_{label}.npy")
            curr_label = np.full((curr_arr.shape[0], 1), label)

            dataset_arr.append(curr_arr)
            label_arr.append(curr_label)


success, parent_data, master_label = shuffle_arrays(dataset_arr, label_arr)
# %% 
# Feature engineering
master_data = log10(parent_data)
global_max = np.nanmax(master_data)
global_min = np.nanmin(master_data)
altered_min = global_min - 1 #normalized 0 reserved for NaN

def try_1(sep):
    """
    Current feature extraction version
    """
    inter = np.multiply(sep, -1)
    inter = shift(inter,1 - np.nanmin(inter))
    inter = np.log2(inter)
    inter = 1 - custom_norm(inter, -1.5, np.nanmax(inter)+ 1.5, 1.)
    return inter

def try_2(master_data):
    master_data = shift(master_data, 2 - global_min)
    master_data = nanlog(master_data, np.log(2)) 
    new_global_max = np.nanmax(master_data)
    master_data = custom_norm(master_data, 0, new_global_max)
    return master_data

def try_3(sep):
    pass

master_data = try_1(master_data)
# %% 
# Print images
plot_data(master_data, master_label, shape = (6, 6), start = 0)

#%%
#reshape data for pca
pca_data = master_data.reshape((master_data.shape[0], master_data.shape[1] * master_data.shape[2]))
# %%
# Apply PCA dim 2 onto images
pca = PCA(n_components=2)
pca.fit(pca_data)
trans_data = pca.transform(pca_data)
# %%
print(pca.components_)
print(pca.explained_variance_)
# %%
plt.scatter(trans_data[:, 0], trans_data[:, 1], c = master_label,alpha=0.5)
plt.colorbar()
# %%
# Attempt manifold learning onto data
mds = MDS(n_components=49, random_state=1)
mds_data = mds.fit_transform(pca_data[0:100])
# %%
plt.scatter(mds_data[:, 0], mds_data[:, 1], c = master_label[0:100],alpha=0.5)
plt.colorbar()
# %%
new_mds_data = mds_data.reshape((100, 7, 7))
plot_data(new_mds_data, master_label, shape = (6, 6), start = 50)
# %%
fit_mds = mds.fit(pca_data[0:100])
np.cumsum(fit_mds.explained_variance_ratio_)
# %%
