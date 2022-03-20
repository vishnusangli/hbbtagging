import matplotlib.pyplot as plt

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