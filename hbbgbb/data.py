# %%
import glob
import sys


import numpy as np
import pandas as pd
import h5py
import itertools
import tqdm
import graph_nets as gn
import tensorflow as tf
import pickle

MODELDIR = 'saved_models'
LOADMODEL = "simplenn"

sys.path.append("..")
import settings
# %%
def load_data(tag='r10201'):
    """
    Load fat jet data into a Dataframe. Basic pre-selection is applied.

    Parameters
    --
        `tag`: str, tag of dataset to use
    """
    # Load the data
    path=glob.glob(f'{settings.datadir}/user.zhicaiz.309450.NNLOPS_nnlo_30_ggH125_bb_kt200.hbbTrain.e6281_s3126_{tag}_p4258.2020_ftag5dev.v0_output.h5/*.output.h5')[0]
    df=pd.read_hdf(path,key='fat_jet')

    # Apply preselection
    df=df[df.nConstituents>2]
    df=df[df.pt>500e3]
    df=df.copy()
    df['mass']=df['mass']/1e3
    df['pt'  ]=df['pt'  ]/1e3

    return df


def general_load(name):
    """
    mass ->  [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 
    1400, 1500, 1600, 1800, 2000, 2250, 2500 (r9364), 2750, 
    3000, 3500, 4000(r9), 5000, 6000 (r9) ]


    """
    path=glob.glob(f'{settings.datadir}/{name}/*.output.h5')[0]
    df=pd.read_hdf(path,key='fat_jet')

    df=df[df.nConstituents>2]
    df=df[df.pt>500e3]
    df=df.copy()
    df['mass']=df['mass']/1e3
    df['pt'  ]=df['pt'  ]/1e3
    return df

def load_newdata(tag = "r10201", sig_mass = ['1000', '1100',
       '1200', '1300', '1400', '1500', '1600', '1800', '2000', '2250',
       '2500', '2750', '3000', '3500', '4000', '5000', '6000'], 
       bag_jx = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']):
    """
    Interleaves signal and background data in b/w \\
    Since model is trained over numerous epochs, I felt there isn't any need to 
    shuffle in between \\
    total lengths: \\
    sig_mass = ['300', '400', '500', '600', '700', '800', '900', '1000', '1100',
       '1200', '1300', '1400', '1500', '1600', '1800', '2000', '2250',
       '2500', '2750', '3000', '3500', '4000', '5000', '6000']

    bag_jx = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    Returns: \\
        signal
    """
    rd = pd.read_csv(f"{settings.hbbgbb_dir}/datafiles.txt")
    rd = np.array(rd.zhicaiz)
    def find_file(part1, tag, rd):
        """
        parses through a txt file of ls and obtains the required file's string
        """
        curr = rd[[part1 in i for i in rd]]
        curr = curr[[tag in i for i in curr]]
        if len(curr) != 1:
            print(f"found {len(curr)} objects matching file filter part {part1}, tag: {tag}")
            return ""
        return curr[0]
    
    signal_arrs = []
    new_sig_mass = []
    for i in tqdm.tqdm(sig_mass):
        try:
            name = find_file(f"_c10_M{i}.", tag, rd)
            if name == "":
                continue
            temp = general_load(name)
            signal_arrs.append(temp)
            new_sig_mass.append(i)
        except IndexError as e:
            print(f"{i} not found in signal for tag {tag}")
            continue
    backg_arrs = []
    new_bag_jx = []
    for i in tqdm.tqdm(bag_jx):
        try:
            name = find_file(f"jetjet_JZ{i}W.", tag, rd)
            temp = general_load(name)
            backg_arrs.append(temp)
            new_bag_jx.append(i)
            if name == "":
                continue
        except IndexError as e:
            print(f"{i} not found in background for tag {tag}")
            continue
    return signal_arrs, backg_arrs, new_sig_mass, new_bag_jx

def label(df):
    """
    Decorate a fat jet `df` DataFrame with labels.
    - `label`: sparese label (0-2)
    - `labelx`: one-hot label `x`
    """
    df['label0']=(df.GhostHBosonsCount==1)
    df['label1']=(df.GhostHBosonsCount==0)&(df.GhostBHadronsFinalCount==2)
    df['label2']=(df.GhostHBosonsCount==0)&(df.GhostBHadronsFinalCount!=2)

    df['label']=3 # default value
    df.loc[df.label0,'label']=0
    df.loc[df.label1,'label']=1
    df.loc[df.label2,'label']=2

def load_data_constit(tag='r10201'):
    """
    Load fat jet constituent data as a `h5py.Dataset`.

    Parameters
    --
        `tag`: str, tag of dataset to use
    """
    path=glob.glob(f'{settings.datadir}/user.zhicaiz.309450.NNLOPS_nnlo_30_ggH125_bb_kt200.hbbTrain.e6281_s3126_{tag}_p4258.2020_ftag5dev.v0_output.h5/*.output.h5')[0]
    f=h5py.File(path, 'r')
    return f['fat_jet_constituents']

def create_graph(fatjet,constit,feat, glob = []):
    """
    Create a dictionary graph for a large R jet. The graph is taken to be fully
    connected. The node features are constituent properties listed in `feat`.

    The `fatjet` is a `pd.Series` with information about the fat jet.

    The `constit` is a structured array with information about the constituents.
    """
    # Global features are properties of the fat jet
    f=['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
    globals = glob


    # Nodes are individual tracks
    nodes=np.array([np.abs(constit[x]) for x in feat]).T

    # Fully connected graph, w/o loops
    i=itertools.product(range(nodes.shape[0]),range(nodes.shape[0]))
    senders=[]
    receivers=[]
    for s,r in i:
        if s==r: continue
        senders.append(s)
        receivers.append(r)
    edges=[[]]*len(senders)

    return {'globals':globals, 'nodes':nodes, 'edges':edges, 'senders':senders, 'receivers':receivers}

def create_graphs(fatjets, constits, feat, glob_vars = None):
    """
    Create fat jet graphs from a list of `fatjets` and their `constits`uents.
    The `feat` is a list of constituent attributes to use as feature nodes.

    The `fatjets` dataframe corresponds to fat jet properties. The index points
    to the entry in `constits` corresponding to that jet.

    The `constits` is a list of structured arrays for all fat jets. Each entry
    contains the information of a constituent.
    """
    dgraphs=[]
    constits=constits[fatjets.index.values]

    for (i,fatjet),constit,gl in tqdm.tqdm(zip(fatjets.iterrows(),constits, glob_vars),total=len(fatjets.index)):
        constit=constit[~np.isnan(constit['pt'])] #IS there an issue here?
        dgraphs.append(create_graph(fatjet, constit, feat, gl))

    return gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)
# %%
def group_create_graphs(fj_list, names, feat, glob_vars = None, tag = 'r10201', type = "signal"):
    """
    Wrapper function for create_graphs that contains and runs for all
    """
    def load_constit(name):
        path=glob.glob(f'{settings.datadir}/{name}/*.output.h5')[0]
        f=h5py.File(path, 'r')
        return f['fat_jet_constituents']

    def find_file(part1, tag, rd):
        """
        parses through a txt file of ls and obtains the required file's string
        """
        curr = rd[[part1 in i for i in rd]]
        curr = curr[[tag in i for i in curr]]
        if len(curr) != 1:
            print(f"found {len(curr)} objects matching file filter part {part1}, tag: {tag}")
            return ""
        return curr[0]
    def open_constit(name):
        """
        Open file with key "fat jet constituents"
        """
        path = glob.glob(f'{settings.datadir}/{name}/*.output.h5')[0]
        f=h5py.File(path, 'r')
        return f
    
    rd = pd.read_csv(f"{settings.hbbgbb_dir}/datafiles.txt")
    rd = np.array(rd.zhicaiz)
    graphs = []
    if type == "signal":
        for i in tqdm.tqdm(range(len(names))):
            name = find_file(f"_c10_M{names[i]}.", tag, rd)
            constit = open_constit(name)
            temp_graph = create_graph(fj_list[i], constit['fat_jet_constituents'], feat, glob_vars)
            constit.close()
            graphs.append(temp_graph)
    else:
        for i in tqdm.tqdm(range(len(names))):
            name = find_file(f"jetjet_JZ{i}W.", tag, rd)
            constit = open_constit(name)
            temp_graph = create_graph(fj_list[i], constit['fat_jet_constituents'], feat, glob_vars)
            constit.close()
            graphs.append(temp_graph)
    
    return graphs



# %%
## Abstract the current system for storing arrays -- usually predictions
def write(name, arr):
    """
    
    """
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(arr, f)

def read(name):
    """
    """
    with open(f"{name}.pkl", 'rb') as infile:
        result = pickle.load(infile)
    return result
# %%
def merge_shuffle(labels, graphs, seed = 0):
    """
    Uses an iterative merge shuffle method that parses 
    through all given arrays to create a master list 

    Each list preserves relative (non-contiguous) ordering

    Paired shuffling method that accordingly does two (label, graph). 
    """
    num_arrs =len(labels)
    
    index_arrs = [0 for i in labels]
    total_size = np.sum([len(a) for a in labels])
    def give_probs(index_arrs):
        curr_size =[len(labels[a]) - index_arrs[a] for a in range(len(labels))]
        size = np.sum(curr_size)
        probs = [np.divide(curr_size[i], size) for i in range(len(curr_size))]
        def here(a, i):
            a[i] += a[i-1]
        [here(probs, i) for i in range(1, len(probs))] 
        return probs
    master_elem = 0
    master_label, master_graphs = [], []
    np.random.seed(seed)
    while np.sum(index_arrs) < total_size:
        probs = give_probs(index_arrs)
        rand = np.random.random()
        for i in range(1000):
            for elem in range(len(probs) - 1, -1, -1):
                if rand <= probs[elem] and index_arrs[elem] < len(labels[elem]):
                    master_label.append(labels[elem][index_arrs[elem]])
                    master_graphs.append(graphs[elem][index_arrs[elem]])
                    index_arrs[elem] += 1
                    master_elem += 1
    return master_label, master_graphs
# %%
def find_file(part1, tag, rd):
    """
    parses through a txt file of ls and obtains the required file's string

    part1 -- 1 filter check
    tag -- tag, another filter check

    rd -- ls dump array
    """
    curr = rd[[part1 in i for i in rd]]
    curr = curr[[tag in i for i in curr]]
    if len(curr) != 1:
        print(f"found {len(curr)} objects matching file filter part {part1}, tag: {tag}")
        return ""
    return curr[0]
# %%

# %%
