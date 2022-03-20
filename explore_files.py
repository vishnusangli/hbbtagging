# %%
import h5py
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hbbgbb import data
from hbbgbb import eng
# %%
def iterate(df_train, fjc_train, first = 'pt', sec = 'trk_d0'):
    """
    Confirm the mismatch in track and calorimeter array sizes
    """
    fjc_train = fjc_train[df_train.index.values]
    for (i, fatjet), constit in tqdm(zip(df_train.iterrows(), fjc_train), total = len(df_train.index)):
        constit = constit[~eng.isnanzero(constit[first])]
        if True in eng.isnanzero(constit[sec]):
            print(f"{i} error")
            break