# %%
#%load_ext autoreload
#%autoreload 2

#%%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kkplot

import hbbgbb.plot as myplt

# %% Arguments
rocs = ['roc_Xbb2020v2.npy', 'roc_Xbb202006.npy']
if 'ipykernel_launcher' not in sys.argv[0]: # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Compare multiple ROC curves')
    parser.add_argument('rocs', type=str, nargs='+', default=rocs, help='Version of the tagger')
    args = parser.parse_args()

    rocs = args.rocs

# %% Load the rocs
therocs={}
labels=set()
for roc in rocs:
    theroc=np.load(roc, allow_pickle=True).item()
    therocs[roc]=theroc
    labels.update(theroc.keys())
labels.remove(0)
colors = ['red', 'blue', 'orange', 'green', 'purple', 'black']
# %% Plot the ROC curves

for label in labels:
    fig, ax = plt.subplots(figsize=(8,6))
    count = 0
    for name, roc in therocs.items():
        ax.plot(roc[0], 1-roc[label], '-', color = colors[count], label=name)
        count += 1
    fig.legend(title='Background')
    ax.set_title('Background: '+myplt.mylabels.get(label,label))
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    kkplot.ticks(ax.xaxis, 0.1, 0.02)
    kkplot.ticks(ax.yaxis, 0.1, 0.02)
    ax.set_xlabel('Signal Efficiency')
    ax.set_ylabel('Background Rejection')
    fig.tight_layout()
    fig.savefig(f'roc_{int(label)}.pdf')
    fig.show()
# %%
