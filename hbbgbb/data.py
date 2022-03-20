import glob
import settings
import numpy as np
import pandas as pd
import h5py
import itertools
import tqdm
import graph_nets as gn

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
    f=h5py.File(path)
    return f['fat_jet_constituents']

def create_graph(fatjet,constit,feat):
    """
    Create a dictionary graph for a large R jet. The graph is taken to be fully
    connected. The node features are constituent properties listed in `feat`.

    The `fatjet` is a `pd.Series` with information about the fat jet.

    The `constit` is a structured array with information about the constituents.
    """
    # Global features are properties of the fat jet
    globals=[]

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

def create_graphs(fatjets, constits, feat):
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
    for (i,fatjet),constit in tqdm.tqdm(zip(fatjets.iterrows(),constits),total=len(fatjets.index)):
        constit=constit[~np.isnan(constit['pt'])] #IS there an issue here?
        dgraphs.append(create_graph(fatjet, constit, feat))

    return gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)