# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hbbgbb import data
from hbbgbb import eng

EXP_OUT = "explore_output"
# %%
mylabels={0:'Higgs',1:'QCD (bb)', 2:'QCD (other)'}
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
        xlabel = "JZ Slice"
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

# %%
mylabels={0:'Higgs',1:'QCD (bb)', 2:'QCD (other)'}
trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
signals = ["1100", "1200", "1400"]
backs = ["5"]
# %% Load Data
print(f"Load Training")
train_loader = data.GraphLoader(signals, backs, graph_dir='ographs')

# %%
batch_size = 50000
batch_g, batch_l = train_loader.give_dict_batch(label_ratio = [0.495, 0.1, 0.495], batch_size=batch_size, hist=False)
labels = np.array(batch_l)
trks = np.array([batch_g[i]["nodes"] for i in range(len(batch_g))])
# %%
### IMPORTANT
num_jets = 2
lens = []
def single_val(arr, feature):
    """
    Reducing the array to a single value
    max, avg, specific element
    """
    lens.append(len(arr))
    return [arr[0]]

def compare_jets(jets, labels, label = 0, unc=False):
    """
    Generate list of singular values for each jet
    """
    num_features = jets[0].shape[1]
    comp_info = [[] for i in range(num_features)] 
    l_jets = jets[labels[:, label]]
    unc_ratios = [[], []] #d0, sintheta
    for i in l_jets:
        temp = np.array(i)
        
        for feat in range(num_features):
            singular_vals = single_val(temp[:, feat], feat)
            [comp_info[feat].append(i) for i in singular_vals]
    comp_info = np.array(comp_info)
    if unc:
        d0_rat = comp_info[0, :]/comp_info[3, :]
        sin_rat = comp_info[1, :]/comp_info[4, :]
        mom_rat = comp_info[0, :]*comp_info[2, :]
        return np.vstack([comp_info, d0_rat, sin_rat, mom_rat])
    return comp_info
#train_data, train_label = data.load_all(train_loader, num_batches = 7)
# %%
info = [compare_jets(trks, labels, label = i, unc = False) for i in range(3)]

more_trk_features = trk_features + ["d0/Unc", "sinTheta/unc"]
# %%
max_lims = [1.795607, 1.9480603, 0.0019965656, 3.344233, 5.662956, 5.3698955, 4.6754017, 0]
min_lims = [1.25, 1.3, 0, 1.3, 1, 1, 1, 0]

max_vals = [0, 0, 0, 0, 0, 0, 0, 0]
min_vals = [0, 0, 0, 0, 0, 0, 0, 0]

outliers = [0, 0, 0, 0, 0, 0, 0, 0]
plt.rcParams.update({'font.size': 14})
num_imgs = len(info[0])
f, ax = plt.subplots(2,4, figsize = (30, 10))
b = ax.flatten()
for l in range(3):
    l_data = info[l]
    for f in range(num_imgs):
        b[f].hist(l_data[f], label = mylabels[l], density = True, histtype = "step", bins = np.linspace(min_lims[f], max_lims[f], 30))
        outliers[f] += int(sum([i > max_lims[f] or i < min_lims[f] for i in l_data[f]]))
        b[f].set_title(f"{more_trk_features[f]}")
        b[f].set_yscale("log")
        b[f].legend()
        
        curr_max = np.max(l_data[f])
        curr_min = np.min(l_data[f])
        if curr_max > max_vals[f]:
            max_vals[f] = curr_max

        if curr_min < min_vals[f]:
            min_vals[f] = curr_min

plt.suptitle(f"All tracks for jets (Batch size: {batch_size})")
plt.savefig(f"{EXP_OUT}/data_stats.pdf")
print(f"Num Outliers for each plot - {outliers}")
print(f"Max {max_vals}")
print(f"Min {min_vals}")
# %%

## any nans in dict
def find_dict_nans(batch_dict):
    places = []
    for d in range(len(batch_dict)):
        if np.any(np.isnan(batch_dict[d]['nodes'])):
            places.append(d)
    return places
###any zeros
def find_dict_zeros(batch_dict):
    places = []
    for d in range(len(batch_dict)):
        if len(np.where(batch_dict[d]['nodes'] == 0)[0]) > 0:
            places.append(d)
    return places

def find_diff_shape(batch_dict):
    '''any nodes that are not shape (2, 7)'''
    places = []
    for d in range(len(batch_dict)):
        if batch_dict[d]['nodes'].shape != (2, 7):
            places.append(d)
    return places
# %%
batch_dict = train_loader.give_dict_batch(hist = False)
find_dict_nans(batch_dict[0])
find_dict_zeros(batch_dict[0])