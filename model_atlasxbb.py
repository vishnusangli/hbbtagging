# %%
#%load_ext autoreload
#%autoreload 2

#%%
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hbbgbb.plot as myplt
import hbbgbb.data as data
from hbbgbb import analysis

# %% Arguments
parser = argparse.ArgumentParser(description='Calculate ROC curves for ATLAS Xbb taggers')
parser.add_argument('version', type=str, nargs='?', default='Xbb202006', help='Version of the tagger')
args = parser.parse_args()

version = args.version

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %% Load the datset
df=data.load_data()
data.label(df)

# %%
f=0.25
df['xbbscore'] = np.log( df[f'{version}_Higgs'] / ((1-f)*df[f'{version}_QCD'] + f*df[f'{version}_Top']) )

# %%
myplt.labels(df,'xbbscore','label',fmt=fmt)
plt.savefig(f'{version}.png')
plt.show()
plt.clf()

# %% Calculate ROC curves
analysis.roc(df, 'xbbscore', f'roc_{version}')
