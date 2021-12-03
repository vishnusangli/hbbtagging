# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
from file_support import *
from model_support import *
import math
import time
import pandas as pd
#from sklearn.model_selection import train_test_split
# %%
#Directory and label file selection
BGROUND_Label = {0:[1]}
M_SIG_Label = {0:[0]}
I_SIG_Label = {0:[2]}
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
            
            curr_label = np.array(pd.read_parquet(f"{DATA_DIR}/{ev_dir[event]}/misc_features.parquet", engine="pyarrow"))
            dataset_arr.append(curr_arr)
            label_arr.append(curr_label)


success, parent_data, misc_vals = shuffle_arrays(dataset_arr, label_arr)
master_label = misc_vals[:, -1]
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
plot_data(master_data, master_label, shape = (5, 4), start = 0)

# %% 
# Train/test split data
train_data, train_label, test_data, test_label = split_data(master_data, master_label, train_ratio=0.8)
# %%
model_predictions = np.load("model_preds_11-23.npy")

# %%
# Generate labels from predictions
# TODO Use f1 metric -- discriminating variable
def disc_var(prediction, f1):
    """
    Discriminating variable for determining predicted variable
    """
    p0 = prediction[0]
    p1 = prediction[1]
    p2 = prediction[2]
    value = np.log(p0/(f1 * p1 + (1 - f1) * p2))
    return value

naive_pred = lambda x: np.array([np.argamx(i) for i in x])
disc_vals = np.array([disc_var(model_predictions[i], 0.9) for i in range(0, model_predictions.shape[0])])

# %%
n, hist_0, hist_1, hist_2 =  generate_labelhist(disc_vals, test_label)
plt_hist(n, hist_0, hist_1, hist_2)