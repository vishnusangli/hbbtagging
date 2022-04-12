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
import hbbgbb.plot as myplt
# %%
output='const_mass'
df = data.load_data()
data.label(df) 

# %% Plot the mass graph
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')
myplt.labels(df, "mass", 'label', fmt=fmt)

# %% Create the bins?
bins = np.linspace(0, 700, 70)
l0_num, b = np.histogram(df[df.label0 == True], bins)
l0_num = l0_num/sum(l0_num)
l1_num, b = np.histogram(df[df.label1 == True], bins)
l1_num = l1_num/sum(l1_num)
l2_num, b = np.histogram(df[df.label2 == True], bins)
l2_num = l2_num/sum(l2_num)
nums = np.vstack((l0_num, l1_num, l2_num)).T

def val(label0, label1, label2):
    const = label0 + (label1/3) + label2
    return [np.divide((label0/0.7), const), np.divide(label1/2, const), np.divide(label2, const)]
probs = []
for (a, b, c) in zip(l0_num, l1_num, l2_num):
    probs.append(val(a, b, c))
probs = np.array(probs)
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
    curr_bin = probs[int(val//70)]
    return np.where( curr_bin == np.max(curr_bin))[0][0]
label_disc = np.vectorize(label_disc)

def spec_label(val, label):
    return probs[int(val//70), label]
spec_label = np.vectorize(spec_label)

def confirm(val, req):
    """
    create logits
    """
    if val == req:
        return 1
    return 0
confirm = np.vectorize(confirm)


df['score0'] = spec_label(df.mass, 0)
df['score1'] = spec_label(df.mass, 1)
df['score2'] = spec_label(df.mass, 2)
df['pred'] = label_disc(df.mass)
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