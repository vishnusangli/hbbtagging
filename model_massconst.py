"""
Provide a pure mass-based classification, based on a set threshold
"""
# %%
import h5py
import sys
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from hbbgbb import data
from hbbgbb import eng
from tqdm import tqdm
from hbbgbb import analysis
# %%
output='const_mass'
df = data.load_data()
data.label(df)

# %%
init_range = [0, 200]

class Discriminator:
    def __init__(self, t):
        self.threshold = t

    def predict(self, x):
        def label_disc(val):
            """
            tagger 
            """
            if val >= self.threshold:
                return 0
            return 2
        label_disc = np.vectorize(label_disc)
        def confirm(val, req):
            """
            create logits
            """
            if val == req:
                return 1
            return 0
        confirm = np.vectorize(confirm)
        pred = pd.DataFrame(label_disc(x.mass), columns = ['pred'])
        pred['score0'] = confirm(pred.pred, 0)
        pred['score1'] = confirm(pred.pred, 1)
        pred['score2'] = confirm(pred.pred, 1)
        return pred



def give_loss(df_fj, t):
    """
    Gives the loss of setting a threshold wherein 
    jets with mass above t are label0 
    """
    pass

# %%
threshold = 90
def label_disc(val):
    """
    tagger 
    """
    if val >= threshold:
        return 0
    return 2
label_disc = np.vectorize(label_disc)

def confirm(val, req):
    """
    create logits
    """
    if val == req:
        return 1
    return 0
confirm = np.vectorize(confirm)

df['pred'] = label_disc(df.mass)
df['score0'] = confirm(df.pred, 0)
df['score1'] = confirm(df.pred, 1)
df['score2'] = confirm(df.pred, 2)
# %%
analysis.roc(df, 'score0', f'roc_{output}')
# %%
"""
use the histogram of mass on the training dataset,
 store it with the respective bins

for each bin --
combined = label0 + (label1/2) + label2
scorex = labelx/combined 
assuming uniform prob -- using normalized to prevent sparse label issues
"""