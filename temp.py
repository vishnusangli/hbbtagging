import h5py
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hbbgbb import data
from hbbgbb import eng

import graph_nets as gn
import time

def det_dist(labels, batch_size = 1000, logits = False):
    """
    Plot the trend of the # of labels across an event 
    with batches of specified size

    Used to display the distribution of the dataset

    logits will have each jet label as [True, False, False ...] in np.array format

    Otherwise a single array of true labels
    """

    num = (len(labels)//batch_size) + 1
    counts = np.zeros(shape = (num, 3))
    if logits:

        for batch in tqdm(range(num)):
            start, end = (batch * batch_size), min(batch_size * (batch + 1), len(labels))
            batch_label = labels[start:end] 
            counts[batch] = [np.sum(labels[start:end, i]) for i in range(3)]
        
    else:
        for batch in tqdm(range(num)):
            start, end = (batch * batch_size), min(batch_size * (batch + 1), len(labels))
            batch_label = labels[start:end]
            counts[batch] = [np.sum(batch_label == 0 ), np.sum(batch_label == 1 ), np.sum(batch_label == 2 )]
        
    batches = range(num)

    plt.figure(figsize= (10, 8))
    plt.yscale('log')
    plt.plot(batches, counts[:, 0], label = 'hbb')
    plt.plot(batches, counts[:, 1], label = 'gbb')
    plt.plot(batches, counts[:, 2], label = 'other')
    plt.ylabel("Count")
    plt.xlabel(f"Batch (Size {batch_size})")
    plt.title("Data Distribution over batches")
    plt.legend()
    plt.savefig("temp.png")
    return counts


trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
signals = ["1400"]
backs = ["5"]
tag = 'r10201'
labels = [0, 1, 2]
strlabels=list(map(lambda l: f'label{l}', labels))
# %%
graphs = []
group_logits = []
# %%
for i in range(len(backs)):
    signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(tag, sig_mass = [], bag_jx=[backs[i]])
    [data.label(i) for i in backg_arrs]
    backg_arrs = [i[np.any(i[strlabels], axis = 1)].copy() for i in backg_arrs]
    if len(new_bag_jx) > 0:
        temp_graphs = data.LoadGraph.master_load(backg_arrs, new_bag_jx, trk_features, tag = tag, type = "back")
        graphs.extend(temp_graphs)
        group_logits.extend([np.array(i[strlabels]) for i in backg_arrs])

# %%
for i in range(len(signals)):
    signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(tag, sig_mass = [signals[i]], bag_jx=[])
    [data.label(i) for i in signal_arrs]
    signal_arrs = [i[np.any(i[strlabels], axis = 1)].copy() for i in signal_arrs]
    if len(new_sig_mass) > 0:
        print("Graphs frunction")
        temp_graphs = data.LoadGraph.master_load(signal_arrs, new_sig_mass, trk_features, tag = tag)
        graphs.extend(temp_graphs)
        group_logits.extend([np.array(i[strlabels]) for i in backg_arrs])

# %%
master_label, master_data = data.merge_shuffle(group_logits, graphs)
master_label = np.array(master_label)
# %%
det_dist(master_label, batch_size = 100000, logits = True)
# %%
t = time.time()
master_data = gn.utils_tf.data_dicts_to_graphs_tuple(master_data)
print(time.time() - t)
# %%
# %%
