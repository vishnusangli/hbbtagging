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
EXP_OUT = "explore_output"
# %%
def iterate(df_train, fjc_train, first = 'pt', sec = 'trk_d0'):
    """
    Confirm the mismatch (zeros) in track arrays with repsect to calorimeter arrays
    for uncharged particles
    """
    fjc_train = fjc_train[df_train.index.values]
    for (i, fatjet), constit in tqdm(zip(df_train.iterrows(), fjc_train), total = len(df_train.index)):
        constit = constit[~eng.isnanzero(constit[first])]
        if True in eng.isnanzero(constit[sec]):
            print(f"{i} error")
            break
# %%
def pt_give_vals(fjc = data.load_data_constit(), ):
    """
    Returns side-by-sied track corresponding calculated track pt to the calorimeter pt

    Each element seems to be off by some relative magnitude. This is because we trk_pt is 
    currently divided by charge as it is. How to find?
    """
    first = fjc[0]
    filter = ~np.isnan(first['pt'])
    features = ['pt', 'trk_qOverP', 'trk_theta']
    df = pd.DataFrame(np.array([first[i][filter] for i in features]).T, columns=features)
    df['mom'] = 1/df['trk_qOverP']
    df['trk_pt'] = df['mom'] * np.sin(df['trk_theta'])
    return df
def pT_Check(df):
    """
    Plot track_pt - calo_pt difference over event dataset

    Uses pt_give_vals output as input
    """
    pass
# %%
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
    return counts

# %%
def stats_wrapper(tag = "r10201"):
    """
   temporary function for easy use 

    """
    sig_mass = ['300', '400', '500', '600', '700', '800', '900', '1000', '1100',
       '1200', '1300', '1400', '1500', '1600', '1800', '2000', '2250',
       '2500', '2750', '3000', '3500', '4000', '5000', '6000']
    bag_jx = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(tag, sig_mass, bag_jx)
    return signal_arrs, backg_arrs, new_sig_mass, new_bag_jx
# %%

def file_stats(fj_list, names, tp = "signal", num_labels = 4, tag = "r10201"):
    """
    Output statistics on Zhicai's hh_bbbb and jetjet qcd event data\\
    Show event sizes and overall label distribution \\

    Imput: Array of fat_jet lists with jet features for a particular type
    """
    labels = ["hbb", "QCD (gbb)", "QCD (other)", "#higgs = 2"]
    [data.label(i) for i in fj_list]
    stats = np.array([[np.sum(fj.label == i) for i in range(num_labels)] for fj in fj_list])
    norm_stats = [[np.divide(i[m], np.sum(i)) for m in range(len(i))] for i in stats]
    norm_stats = np.array(norm_stats)

    if tp == "signal":
        xlabel = "Mass"
        name = "signal"
    else:
        xlabel = "Jx"
        name = "jetjet"

    ## Regular Plot
    for p in range(2):
        plt.figure(figsize= (10, 8))
        prev = np.zeros(len(stats))
        for i in range(num_labels):
            plt.bar(names, stats[:, i], bottom = prev, label = labels[i])
            prev += stats[:, i]
        plt.title(f" Size of {name} events")
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.legend()
        if p == 0:
            plt.savefig(f"{EXP_OUT}/count_reg.pdf")
        else:
            plt.yscale("log")
            plt.savefig(f"{EXP_OUT}/count_log.pdf")

    ## Norm plot
    plt.figure(figsize= (10, 8))
    prev = np.zeros(norm_stats.shape[0])
    for i in range(num_labels):
        plt.bar(names, norm_stats[:, i], bottom = prev, label = labels[i])
        prev += norm_stats[:, i]
    plt.title(f" Size of {name} events")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"{EXP_OUT}/count_norm.pdf")





def temp():
    """
    Temp function for easy use with copy-paste calls
    """
    features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']

    signals = ["1100", "1200", "1400"]
    backs = ["6", "7", "8"]
    tag = 'r10201'
    signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(sig_mass = signals, bag_jx=backs, tag = tag)

    sig_graphs = data.group_create_graphs(signal_arrs, new_sig_mass, features)
    back_graphs = data.group_create_graphs(backg_arrs, new_bag_jx, features, type = "back")
    return sig_graphs, back_graphs, signal_arrs, backg_arrs, (new_sig_mass, new_bag_jx)
# %%
#Creating the Feature dataset for Feature NN
#Params and features
trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
signals = ["1100", "1200", "1400"]
backs = ["6", "7", "8"]
tag = 'r10201'
labels = [0, 1, 2]
strlabels=list(map(lambda l: f'label{l}', labels))

#Load Data
signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(sig_mass = signals, bag_jx=backs, tag = tag)
[data.label(i) for i in signal_arrs]
[data.label(i) for i in backg_arrs]

#Remove label 3/only required data
signal_arrs = [i[np.any(i[strlabels], axis = 1)].copy() for i in signal_arrs]
backg_arrs = [i[np.any(i[strlabels], axis = 1)].copy() for i in backg_arrs]

# Combine and shuffle dataset into one
master_arr = signal_arrs + backg_arrs
master_data = [np.array(i[calo_features]) for i in master_arr]
master_label = [np.array(i[strlabels]) for i in master_arr]
master_label, master_data = data.merge_shuffle(master_label, master_data)

# %%
#Creating 1 graph for GraphNN
trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
signals = ["1100", "1200", "1400", "1600"]
backs = ["6", "7"]
tag = 'r10201'
labels = [0, 1, 2]
strlabels=list(map(lambda l: f'label{l}', labels))

#Load fj data
signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(sig_mass = signals, bag_jx=backs, tag = tag)
[data.label(i) for i in signal_arrs]
[data.label(i) for i in backg_arrs]



