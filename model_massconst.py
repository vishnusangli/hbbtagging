"""
Provide a pure mass-based classification, based on a set threshold
"""
# %%
import h5py

import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from hbbgbb import data
from hbbgbb import eng
from tqdm import tqdm
# %%
df = data.load_data()
data.label(df)
# %%
init_range = [0, 200]

def give_loss(df_fj, t):
    """
    Gives the loss of setting a threshold wherein 
    jets with mass above t are label0 
    """
    pass

def get_preds(df_fj, t):
    preds = pd.DataFrame()
    pass