import matplotlib.pyplot as plt

import pandas as pd
from numpy import argmax
mylabels={0:'Higgs',1:'QCD (bb)', 2:'QCD (other)'}
histcommon={'histtype':'step','density':True}

def labels(df, varname, labelcol, predcol=None, fmt=None, ax=None):
    if ax is None:
        ax=plt.gca()

    histargs=fmt.hist(varname) if fmt is not None else {}
    histargs['density']=True

    for labelidx in sorted(mylabels.keys()):
        # Plot correctly labeled thing
        sdf=df[df[labelcol]==labelidx]
        _,_,patch=ax.hist(sdf[varname],
                    label=mylabels[labelidx] if predcol is None else None,
                    linestyle='--',
                    **histargs)

        # Plot the predicted thing
        if predcol is not None:
            sdf=df[df[predcol]==labelidx]
            ax.hist(sdf[varname],
                        label=mylabels[labelidx],
                        color=patch[0].get_edgecolor(),
                        **histargs)

    ax.legend()

    fmt.subplot(varname, ax=ax)
    ax.set_ylabel('normalized')

def get_current_graph(logits, true_labels, output, label_conv, fmt = None, curr_label = 0, ax = None):
    """
    Wrapper function that handles plotting of a single score distribution
    subplot in an epoch?
    """
    df = pd.DataFrame(logits, columns = [f'score{i}' for i in labels])
    df['label'] = argmax(true_labels, axis = 1)
    labels(df,f'score{curr_label}','label',fmt=fmt, ax=ax)
    ax.set_title(f"model {output} label{curr_label} - {label_conv[curr_label]}")