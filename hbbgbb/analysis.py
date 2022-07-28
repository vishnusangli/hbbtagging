import numpy as np
import matplotlib.pyplot as plt
import sys
import kkplot

from . import plot as myplt
from sklearn.metrics import auc

ROC_DIR = "roc_curves"
ROC_IMG = "roc_imgs"
def roc(df, score, output=None, plot=True):
    """
    Create ROC curves given `score` column.

    The return value is a dictionary with key `label#` and value the CDF of the
    score for that label. 

    Optional output is also supported as npy (`output="fileprefix"`) or
    plots (`plot==True`, saved to `output`).
    """
    labels=sorted(df['label'].unique())

    # Calculate ROC curves
    rocs={}

    mymin=np.floor(df[score].min())
    mymax=np.ceil(df[score].max())

    for label in labels:
        h,b=np.histogram(df[df.label==label][score],bins=100,range=(mymin,mymax))
        h=1-np.cumsum(h)/np.sum(h) # turn into CDF
        rocs[label]=h

    # Plot ROC curves
    if plot:
        fig, ax=plt.subplots(1,1,figsize=(8,6))
        for label in labels:
            if label==0: continue # this is signal
            ax.plot(rocs[0],1-rocs[label],'-',label=myplt.mylabels.get(label,label))
    
        ax.set_xlabel('Signal Efficiency')
        ax.set_ylabel('Background Rejection')
        kkplot.ticks(ax.xaxis, 0.1, 0.02)
        kkplot.ticks(ax.yaxis, 0.1, 0.02)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        fig.legend(title='Background')
        fig.tight_layout()
        if output is not None:
            fig.savefig(f"{ROC_DIR}/{ROC_IMG}/{output}.pdf")
        fig.show()
        plt.close(fig)
    # Save ROC curves
    if output is not None:
        np.save(f'{ROC_DIR}/{output}.npy',rocs)

def bare_roc(preds, true_labels:np.array, score:int, output:str = None, epoch_roc = False, epoch_num = 0) -> None:
    """
    roc function with separate inputs, not in pd Dataframe

    with respect to the score elem number

    Otherwise a copy
    """
    labels = preds.shape[1]
    rocs={}
    mymin=np.floor(preds[:, score].min())
    mymax=np.ceil(preds[:, score].max())

    for label in range(labels):
        h,b=np.histogram(preds[true_labels==label, score],bins=100,range=(mymin,mymax))
        h=1-np.cumsum(h)/np.sum(h) # turn into CDF
        rocs[label]=h

    fig, ax=plt.subplots(1,1,figsize=(8,6))
    for label in range(labels):
        if label==0: continue # this is signal
        ax.plot(rocs[0],1-rocs[label],'-',label=myplt.mylabels.get(label,label))
    ax.set_xlabel('Signal Efficiency')
    ax.set_ylabel('Background Rejection')
    kkplot.ticks(ax.xaxis, 0.1, 0.02)
    kkplot.ticks(ax.yaxis, 0.1, 0.02)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.legend(title='Background')
    fig.tight_layout()

    if not epoch_roc:
        if output is not None:
            fig.savefig(f"{ROC_DIR}/{ROC_IMG}/{output}.pdf")
        fig.show()
        if output is not None:
            np.save(f'{ROC_DIR}/{output}.npy',rocs)
    else:
        np.save(f'epochs/{output}_epoch-{epoch_num}.npy',rocs)
        fig.savefig(f"model_stats/roc_curve.png")
    plt.close(fig)
    
def aoc(preds, real_labels, score = 0):
    """
    Avoiding the use of appending to data pd
    Seems to be returning different values
    """
    labels = [0, 1, 2]

    rocs = {}
    mymin = np.floor( np.min(preds[:, score]))
    mymax = np.ceil(np.max(preds[:, score]))
    for label in labels:
        h, b = np.histogram(preds[real_labels == label, score], bins = 100, range = (mymin, mymax))
        h = 1- np.cumsum(h)/np.sum(h)
        rocs[label] = h
    
    area = [1- auc(rocs[0], 1- rocs[label]) for label in labels]
    return area
    
    


    
