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
labels = [0, 1, 2]
file = 1100
signal = 1
tag = 'r10201'

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
def alt_create_graphs(fatjets, constits, feat):
    """
    This is a near-copy of data.create_graphs 

    Create fat jet graphs from a list of `fatjets` and their `constits`uents.
    The `feat` is a list of constituent attributes to use as feature nodes.

    The `fatjets` dataframe corresponds to fat jet properties. The index points
    to the entry in `constits` corresponding to that jet.

    The `constits` is a list of structured arrays for all fat jets. Each entry
    contains the information of a constituent.
    """
    graphs =[[], [], []]
    constits=constits[fatjets.index.values]

    for (i,fatjet),constit in tqdm.tqdm(zip(fatjets.iterrows(),constits),total=len(fatjets.index)):
        constit=constit[~np.isnan(constit['pt'])] #IS there an issue here?
        label = fatjet.label
        graphs[label].append(data.create_graph(fatjet, constit, feat))

    return graphs


# %%
## Load Files
if signal:
    filename = data.find_file(f"_c10_M{file}.", tag)
else:
    filename = data.find_file(f"jetjet_JZ{file}W.", tag)

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

# %%
## Method 2
constits = np.array(constits)
constits = np.dstack([constits[i][fatjets.index.values] for i in trk_features])
graphs = data.LoadGraph.biased_single_event(fatjets, constits, trk_features)
# %%
## Naming convention - 
## (tag)_(hh_bbbb/jetjet)_(name)
## label_().pkl
if signal:
    part1 ="hh_bbbb"
else:
    part1 = "jetjet"
dir = f'{settings.graphs}/{tag}_{part1}_{file}'
if not os.path.exists(dir):
    os.makedirs(dir)

for i in range(3):
    with open(f"{dir}/label_{i}.pkl", 'wb') as f:
        pickle.dump(graphs[i], f)

# %%
f.close()

# %%