import numpy as np
import matplotlib.pyplot as plt
import sys
import kkplot

from . import plot as myplt
from sklearn.metrics import auc

ROC_DIR = "roc_curves"
ROC_IMG = "roc_imgs"

def custom_div(x, y):
    if x == 0: return 0
    if y == 0: return x * np.inf
    return np.divide(x, y)
custom_div = np.vectorize(custom_div)

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
        h=1- custom_div( np.cumsum(h),np.sum(h)) # turn into CDF
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

def bare_roc(preds, true_labels:np.array, score:int, output:str = None, epoch_roc = False, epoch_num = 0, file_dir = 'model_stats') -> None:
    """
    roc function with separate inputs, not in pd Dataframe

    with respect to the score elem number

    Otherwise a copy
    """
    labels = 3 #preds.shape[1]
    # For ease, I am forcing the 3-label standard
    rocs={}
    mymin=np.floor(preds[:, score].min())
    mymax=np.ceil(preds[:, score].max())

    for label in range(labels):
        temp = preds[true_labels==label, score]
        if len(temp) > 0:
            h,b=np.histogram(temp,bins=100,range=(mymin,mymax))
        else:
            h = np.tile(np.NaN, 100)
        h=1- custom_div(np.cumsum(h),np.sum(h)) # turn into CDF
        rocs[label]=h

    fig, ax=plt.subplots(1,1,figsize=(8,6), facecolor = "white")
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
        ax.set_title(f"{output} ROC curve")
        fig.savefig(f"{file_dir}/roc_curve.png")
    plt.close(fig)

def aoc(preds, real_labels, score = 0, labels_used = [0, 1, 2]):
    """
    Avoiding the use of appending to data pd
    Seems to be returning different values
    """
    labels = labels_used

    rocs = {}
    mymin = np.floor( np.min(preds[:, score]))
    mymax = np.ceil(np.max(preds[:, score]))
    for label in labels:
        temp = preds[real_labels == label, score]
        if len(temp) > 0:
            h,b=np.histogram(temp,bins=100,range=(mymin,mymax))
        else:
            h = np.tile(0, 100)
        h = 1- custom_div( np.cumsum(h),np.sum(h))
        rocs[label] = h

    area = [1 - auc(rocs[score], 1- rocs[label]) for label in labels]

    to_return = area
    return to_return

    
    


    
