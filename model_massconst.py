# %%
import h5py
import sys

import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

from hbbgbb.models import SimpleModel
import settings
import glob
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