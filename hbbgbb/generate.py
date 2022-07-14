# %%
import glob
import sys
import os

import numpy as np
import pandas as pd
import h5py
import itertools
import tqdm
import graph_nets as gn
import tensorflow as tf
import pickle
import networkx
import data
from itertools import combinations
import graph_nets as gn

sys.path.append("..")
import settings
# %%
trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
labels = [0, 1, 2]
file = 1100
signal = 1
tag = 'r10201'
maxnum = 300000

DIRECTORY = 'labeled_graphs'

if 'ipykernel_launcher' not in sys.argv[0]: # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Train GNN from track features')
    parser.add_argument('--file', type=int, default=file, help='sample number')
    parser.add_argument('--signal', type=int, default=signal, help='signal or background')
    parser.add_argument('--tag', type=str, default=tag, help='tag')

    args = parser.parse_args()

    file = args.file
    signal = args.signal
    tag = args.tag

strlabels=list(map(lambda l: f'label{l}', labels))
# %%
## Load Files
if signal:
    filename = data.find_file(f"_c10_M{file}.", tag)
    part1 = f"hh_bbbb"
else:
    filename = data.find_file(f"jetjet_JZ{file}W.", tag)
    part1 = f"jetjet"

f = data.load_fjc(filename)
constits = f['fat_jet_constituents']
fatjets = data.general_load(filename)
# %%
## Labelling 
data.label(fatjets)
fatjets = fatjets[np.any(fatjets[strlabels], axis = 1)].copy()

# %%
## Method 1
#graphs = alt_create_graphs(fatjets, constits, trk_features)



def biased_single_event(fatjets, constits, feat):
    dgraphs = [[], [], []]
    real_iter = 0
    for (i,fatjet),constit in tqdm.tqdm(zip(fatjets.iterrows(),constits),total=len(fatjets.index)):
        constit=constit[~data.isnanzero(constit[:, 0])] #Use first column as indicator
        label = fatjet.label
        success, graph_dict = data.LoadGraph.create_single_graph(fatjet, constit, feat, num_jets=100)
        ### the above function call generates a single graph #####
        if success:
            dgraphs[label].append(graph_dict) #Number of jets
        real_iter += 1
        if real_iter >maxnum:
            break
    return dgraphs

# %%

## Method 2
# Filters required features and performs absolute of all valeus
constits = np.array(constits)
constits = np.dstack([constits[i][fatjets.index.values] for i in trk_features])
# %%
## Feature Normalization (A manually-set bound that focuses 
# on the distribution and excludes outliers)

constits = np.abs(constits)
norm_lims = [2, 3, 0.00175, 5, 10]
for col in range(len(norm_lims)):
    constits[:, :, col] = np.divide(constits[:, :, col], norm_lims[col])

# %%
# Uncertainty ratios
def nan_divide(x, y):
    if x == 0. or y == 0.:
        return 0
    return np.divide(x, y)
nan_divide = np.vectorize(nan_divide)
d0_unc = nan_divide(constits[:, :, 0], constits[:, :, 3])
d0_unc = d0_unc /210
sthet_unc = nan_divide(constits[:, :, 1], constits[:, :, 4])
sthet_unc = sthet_unc/125
constits = np.dstack([constits, d0_unc, sthet_unc])

# %%
graphs = biased_single_event(fatjets, constits, trk_features)


# %%
## Naming convention - 
## (tag)_(hh_bbbb/jetjet)_(name)
## label_().pkl
if signal:
    part1 ="hh_bbbb"
else:
    part1 = "jetjet"
dir = f'{DIRECTORY}/{tag}_{part1}_{file}'
if not os.path.exists(dir):
    os.makedirs(dir)

for i in range(3):
    with open(f"{dir}/label_{i}.pkl", 'wb') as f:
        pickle.dump(graphs[i], f)

# %%
f.close()
# %%
'''
import plot as myplt
import matplotlib.pyplot as plt
import formatter
fmt=formatter.Formatter('../variables.yaml')

statsdir = f'{settings.statsdir}/{tag}_{part1}_{file}'
if not os.path.exists(statsdir):
    os.makedirs(statsdir)

### Fat jet stats
for feature in calo_features+['nConstituents']:
   myplt.labels(fatjets, feature, 'label', fmt=fmt)
   plt.title(f"{feature} for {part1}_{file}")
   plt.savefig(f'{statsdir}/labels_{feature}.pdf')
   plt.show()
   plt.clf()
'''
# %%
### fjc data


# %%
"""
Current files used for 
signals: 1100, 1200
background: 5

python generate.py --file - --signal  1 --file r9364

"""