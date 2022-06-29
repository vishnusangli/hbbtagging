# %%
import glob
from symtable import SymbolTable
import sys


import numpy as np
import pandas as pd
import h5py
import itertools
import tqdm
import graph_nets as gn
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

MODELDIR = 'saved_models'
LOADMODEL = "simplenn"

sys.path.append("..")
import settings
from hbbgbb import plot as myplot
# %%
### READING EVENT DATA ###
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
def isnanzero(x):
    return np.isnan(x) or x == 0.
isnanzero = np.vectorize(isnanzero)

### GENERATING GRAPHS ###
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
def group_create_graphs(fj_list:list, names:list, feat:list, glob_vars:list = None, tag:str = 'r10201', type:str = "signal", fil_col: str = 'pt'):
    """
    Parent function that generates graphs for numerous events of a particular type (signal, background) \\
    Opens fat jet constituents and obtains track data, using fat jet feature input to create graphs


    Currently outputs a general array of arrays of all graphs per event. Would need to be evenly shuffled b/w 
    signal & background before created into graphs_tuple data structure
    """
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
    
    def single_graph(fatjet, constit, feat, glob_vars):
        globals = glob_vars

        # Nodes are individual tracks
        nodes=np.array(constit)

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
    
    def single_event_create(fatjets, constits, feat, glob_vars = None):
        """
        Sub function that appends all event-generated graphs to the graph_arr.
        Will convert the entire dataset to a Graph once all graphs from all events are appended

        Otherwise identical to create_graphs
        """
        #constits=constits[fatjets.index.values] #bottleneck
        dgraphs = []
        for (i,fatjet),constit,gl in tqdm.tqdm(zip(fatjets.iterrows(),constits, glob_vars),total=len(fatjets.index)):
            constit=constit[~np.isnan(constit[:, 0])] #Use first column as indicator
            dgraphs.append(single_graph(fatjet, constit, feat, gl))
        return dgraphs
    
    rd = pd.read_csv(f"{settings.hbbgbb_dir}/datafiles.txt")
    rd = np.array(rd.zhicaiz)
    graphs = []
    for i in range(len(names)):
        print(names[i])
        if type == "signal":
            name = find_file(f"_c10_M{names[i]}.", tag, rd)
        else:
            name = find_file(f"jetjet_JZ{names[i]}W.", tag, rd)
        # Read constit file
        path = glob.glob(f'{settings.datadir}/{name}/*.output.h5')[0]
        constit=h5py.File(path, 'r')

        #Convert to np array
        use_constit = np.array(constit['fat_jet_constituents'])

        # Zip to each track's features (num events, 100, track chars)
        temp_arr = np.dstack([use_constit[i][fj_list[i].index.values] for i in feat])
        temp_graph = single_event_create(fj_list[i], temp_arr, feat, glob_vars)
        constit.close()
        graphs.append(temp_graph)
    return graphs
    
class LoadGraph:
    def master_load(fj_list:list, names:list, feat:list, glob_vars:list = [], tag:str = 'r10201', type:str = "signal") -> list:
        graphs = []

        for i in range(len(names)):
            print(names[i])
            curr_fj = fj_list[i]
            if type == "signal":
                name = find_file(f"_c10_M{names[i]}.", tag)
            else:
                name = find_file(f"jetjet_JZ{names[i]}W.", tag)
            path = glob.glob(f'{settings.datadir}/{name}/*.output.h5')[0]
            f=h5py.File(path, 'r')
            constits = np.array(f['fat_jet_constituents'])

            constits = np.dstack([constits[i][curr_fj.index.values] for i in feat])
            temp_graph = LoadGraph.gen_single_event(curr_fj, constits, feat, glob_vars)
            f.close()
            graphs.append(temp_graph)
        return graphs

    def create_single_graph(fatjet, constit, feat, glob_vars = [], num_jets = None):
        """
        Generation of a single graph_dict. \n
            `fatjet`: Dataframe row of jet
            `constit`: np 2d array of track info
            `feat`: track features
            `glob_vars`: init global features. These are altered throughout the function
            `num_jets`: upper bound on the number of jet

        If number of jets is less than or equal to 1, graph creation is forfeited.
        """
        glob_vars.append(fatjet['mass'])
        globals = glob_vars
        #fj_vars = ['Xbb2020v2_QCD', 'Xbb2020v2_Higgs', 'Xbb2020v2_Top']
        #for elem in fj_vars:
        #    globals.append(fatjet[elem])

        # Nodes are individual tracks
        nodes=np.array(constit)
        if len(nodes) <= 1:
            return False, []
        if num_jets != None and nodes.shape[0] >= num_jets: 
            nodes = nodes[:num_jets]

        # Fully connected graph, w/o loops
        i=itertools.product(range(nodes.shape[0]),range(nodes.shape[0]))
        senders=[]
        receivers=[]
        for s,r in i:
            if s==r: continue
            senders.append(s)
            receivers.append(r)
        edges=[[]]*len(senders)

        return True, {'globals':globals, 'nodes':nodes, 'edges':edges, 'senders':senders, 'receivers':receivers}

    def gen_single_event(fatjets, constits, feat, glob_vars = []):
        """
        Sub function that appends all event-generated graphs to the graph_arr.
        Will convert the entire dataset to a Graph once all graphs from all events are appended
        Otherwise identical to create_graphs
        """
        #constits=constits[fatjets.index.values] #bottleneck
        dgraphs = []
        if len(glob_vars) != len(fatjets):
            for (i,fatjet),constit in tqdm.tqdm(zip(fatjets.iterrows(),constits),total=len(fatjets.index)):
                constit=constit[~np.isnan(constit[:, 0])] #Use first column as indicator
                dgraphs.append(LoadGraph.create_single_graph(fatjet, constit, feat))
        else:
            for (i,fatjet),constit, gl in tqdm.tqdm(zip(fatjets.iterrows(),constits, glob_vars),total=len(fatjets.index)):
                constit=constit[~np.isnan(constit[:, 0])] #Use first column as indicator
                dgraphs.append(LoadGraph.create_single_graph(fatjet, constit, feat, gl))

        return dgraphs

    def biased_single_event(fatjets, constits, feat):
        dgraphs = [[], [], []]
        for (i,fatjet),constit in tqdm.tqdm(zip(fatjets.iterrows(),constits),total=len(fatjets.index)):
            constit=constit[~isnanzero(constit[:, 0])] #Use first column as indicator
            label = fatjet.label
            dgraphs[label].append(LoadGraph.create_single_graph(fatjet, constit, feat))
        return dgraphs
# %%
### STORING AND READING FILES ###

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
        for i in range(100):
            rand = np.random.random()
            for elem in range(len(probs)):
                if rand <= probs[elem] and index_arrs[elem] < len(labels[elem]):
                    master_label.append(labels[elem][index_arrs[elem]])
                    master_graphs.append(graphs[elem][index_arrs[elem]])
                    index_arrs[elem] += 1
                    master_elem += 1
    return master_graphs, master_label
# %%
def find_file(part1, tag):
    """
    parses through a txt file of ls and obtains the required file's string

    part1 -- 1 filter check
    tag -- tag, another filter check

    rd -- ls dump array
    """
    rd = pd.read_csv(f"{settings.hbbgbb_dir}/datafiles.txt")
    rd = np.array(rd.zhicaiz)
    curr = rd[[part1 in i for i in rd]]
    curr = curr[[tag in i for i in curr]]
    if len(curr) != 1:
        print(f"found {len(curr)} objects matching file filter part {part1}, tag: {tag}")
        return ""
    return curr[0]

def load_fjc(name):
    path = glob.glob(f'{settings.datadir}/{name}/*.output.h5')[0]
    constit=h5py.File(path, 'r')
    return constit

class GraphLoader:
    def __init__(self, signals, backgrounds, tag = 'r10201', labels = 3, graph_dir = settings.graphs) -> None:
        """
        Take label 0 from signal
        use weighted func to take from backs
        1 of each at a tmie
        """
        self.graph_dir = graph_dir
        self.tag = tag
        self.num_labels = labels
        temp = self.combine(signals, "hh_bbbb") + self.combine(backgrounds, "jetjet")
        self.available = [self.combine(signals, "hh_bbbb"), temp, temp.copy()]
        self.finished = False

        self.dict_files = [None for i in range(labels)]
        for i in range(labels):
            self.refill_label(i)

    def combine(self, files, part):
        temp = [part] * len(files)
        return list(zip(files.copy(), temp))
    def load_file(self, filename):
        with open(filename, 'rb') as f:
            list_of_dicts = pickle.load(f)
        return list_of_dicts
    
    def fill_data(self, label, file, sampletype = "hh_bbbb"):
        file = f"hbbgbb/{self.graph_dir}/{self.tag}_{sampletype}_{file}/label_{label}.pkl"
        dicts = self.load_file(f"{file}")
        self.dict_files[label] = [dicts, 0]

    def refill_label(self, label):
        label_available = self.available[label]
        if len(label_available) > 0:
                file = label_available.pop()
                self.fill_data(label, file[0], sampletype = file[1])
        else:
            print(f"Samples for label {label} depleted")
            self.finished = True
        
    def give_dict_batch(self, label_ratio = [0.47, 0.06, 0.47], batch_size = 10000, hist = True, 
    trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']):
        nums = [int(i * batch_size) for i in label_ratio]
        labels, graphs = [], []
        for i in range(len(nums)):
            curr_graphs, curr_labels = self.give_data(i, nums[i])
            labels.append(curr_labels)
            graphs.append(curr_graphs)
        if self.finished:
            return [], []
        g, l = merge_shuffle(labels, graphs)

        if hist:
            display_batch_jets(np.array(labels), np.array([g[i]["nodes"] for i in range(len(graphs))]))
        return g, l


    def give_batch(self, label_ratio = [0.47, 0.06, 0.47], batch_size = 10000):
        g, l = self.give_dict_batch(label_ratio, batch_size, hist = False)
        if len(l) > 0:
            return gn.utils_tf.data_dicts_to_graphs_tuple(g), l
        else:
            return [], []

    def give_data(self, label, num_jets, lab = True):
        """
        Return a given label's jets and change iter
        """
        ##Assume it is length 1 right now
        
        label_arr = self.dict_files[label]
        start, end = label_arr[1], label_arr[1] + num_jets
        if self.finished:
            return [], []

        if len(label_arr[0]) < end:
            dicts = label_arr[0][start: len(label_arr[0])]
            diff = end - len(label_arr[0])
            self.dict_files[label] = None
            self.refill_label(label)
            if not self.finished:
                dicts_new, t = self.give_data(label, diff, False)
                dicts = dicts + dicts_new
            else:
                num_jets = num_jets - diff
        else:
            dicts = label_arr[0][start:end]
            label_arr[1] = end

        if not lab:
            num_jets = 0
        return dicts, self.gen_logits(label, num_jets)

    def gen_logits(self, label, size):
        if size == 0:
            return []
        
        temp = [np.ones(size, dtype = int) * i for i in range(3)]
        temp = [i == label for i in temp]
        return np.dstack(temp)[0]

    def is_finished(self):
        return self.finished

def load_all(loader: GraphLoader, batch_size: int = 10000, num_batches:int = 200):
    """
    Load all batches of a given dataset. Returned as a list of np arrays for 
    batch graphs and labels
    """
    total_g, total_l = [], []
    num = 1
    while not loader.is_finished():
        if num > num_batches: break
        print(f"Batch {num}")
        batch_g, batch_l = loader.give_batch(label_ratio = [0.497, 0.06, 0.497], batch_size=batch_size)
        if len(batch_l) > 0:
            total_g.extend([batch_g])
            total_l.extend([np.array(batch_l)])

        num += 1
    return total_g, total_l

def display_batch_jets(labels, trks, trk_features= ['trk_btagIp_d0',
'trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty'], single_func = None):
    """
    
    """
    def single_val(arr):
        """
        Reducing the array to a single value
        max, avg, specific element
        """
        return np.average(arr)

    if single_func == None:
        single_func = single_val
    def compare_jets(jets, labels, label = 0):
        """
        Generate list of singular values for each jet
        """
        num_features = jets[0].shape[1]
        comp_info = [[]] * num_features
        l_jets = jets[labels[:, label]]
        for i in l_jets:
            [comp_info[m].append(single_func(i[:, m])) for m in range(num_features)]
        return comp_info

    info = [compare_jets(trks, labels, label = i) for i in range(3)]
    f, ax = plt.subplots(len(trk_features)//2,1 + len(trk_features)//2, figsize = (12, 8))
    b = ax.flatten()
    for l in range(3):
        l_data = info[l]
        for f in range(len(trk_features)):
            b[f].hist(l_data[f], label = myplot.mylabels[l], range = (0, 0.2), bins = 30, density = True, histtype = "step")
            b[f].set_title(f"{trk_features[f]}")
            b[f].legend()