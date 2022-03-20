# %%
#%load_ext autoreload
#%autoreload 2

# %%
from re import M
import sys
from sklearn.covariance import MinCovDet

import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import matplotlib.pyplot as plt

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis
from hbbgbb.models import graphs

import settings
STATSDIR = 'data_stats'
MODELSTATS = 'model_stats'
# %% Arguments
features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta']
labels=[0,1,2]
output='graph'
epochs=10

if 'ipykernel_launcher' not in sys.argv[0]: # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Train GNN from track features')
    parser.add_argument('features', nargs='*', default=features, help='Features to train on.')
    parser.add_argument('--output', type=str, default=output, help='Output name.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train.')
    args = parser.parse_args()

    features = args.features
    output = args.output
    epochs = args.epochs

strlabels=list(map(lambda l: f'label{l}', labels))

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %% Load per jet information
df_train=data.load_data()
data.label(df_train)

df_test=data.load_data('r9364')
data.label(df_test)

# %% Filter only specific labels
df_train=df_train[np.any(df_train[strlabels],axis=1)].copy()
df_test =df_test [np.any(df_test [strlabels],axis=1)].copy()

# %% Create tensors of labels
l_train=tf.convert_to_tensor(df_train[strlabels])
l_test =tf.convert_to_tensor(df_test [strlabels])

# %% Load jet constituent data
fjc_train=data.load_data_constit()
g_train=data.create_graphs(df_train, fjc_train,features)

fjc_test=data.load_data_constit('r9364')
g_test =data.create_graphs(df_test , fjc_test ,features)

#%% pltting code
# gs=gn.utils_np.graphs_tuple_to_data_dicts(g_train)
# ls=l_train.numpy()
# df=pd.concat([pd.DataFrame({f'f{i}':g['nodes'][:,i] for i in range(g['nodes'].shape[1])}|{'l':[l[0]]*g['nodes'].shape[0]}) for g,l in zip(gs,ls)])

# #%%
# fig,ax=plt.subplots(1,1,figsize=(8,8))
# for col in df.columns:
#     if not col.startswith('f'): continue
#     ax.clear()
#     b=100
#     for l0, sdf in df.groupby('l'):
#         _,b,_=ax.hist(sdf[col],bins=b,label=f'{l0}',histtype='step')
#     ax.set_xlabel(col)
#     ax.set_yscale('log')
#     ax.legend(title='label0')
#     fig.savefig(col)

# %% Training procedure
class Trainer:
    def __init__(self, model):
        # Model to keep track of
        self.model= model

        # Training tools
        self.stat = pd.DataFrame(columns=['train_loss','test_loss'])
        self.opt  = snt.optimizers.Adam(learning_rate=0.1)

    def step(self, graphs, labels, g_test=None, l_test=None):
        """Performs one optimizer step on a single mini-batch."""
        # Write test data
        test_loss=0.
        if g_test is not None:
            pred = self.model(g_test)
            logits=pred.globals
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=l_test)
            test_loss = tf.reduce_mean(loss)

        # Training
        with tf.GradientTape() as tape:
            pred = self.model(graphs)
            logits=pred.globals
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
            loss = tf.reduce_mean(loss)

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.opt.apply(grads, params)

        # save training status
        self.stat=self.stat.append({'train_loss':float(loss), 'test_loss':float(test_loss)}, ignore_index=True)

        return loss

# %% Prepare for training
model = graphs.INModel(len(labels),2)
t = Trainer(model)

# %% Training
fig_s,ax_s=plt.subplots(ncols=3,figsize=(24,8))
fig_t,ax_t=plt.subplots(figsize=(8,8))

for epoch in tqdm.trange(epochs):
    loss=float(t.step(g_train,l_train, g_test, l_test))

    # Plot the status of the training
    ax_t.clear()
    ax_t.plot(t.stat.train_loss,label='Training')
    ax_t.plot(t.stat.test_loss ,label='Test')
    ax_t.set_yscale('log')
    ax_t.set_ylabel('loss')
    ax_t.set_ylim(1e-1, 1e3)
    ax_t.set_xlabel('epoch')
    ax_t.legend()
    fig_t.savefig(f'{MODELSTATS}/training.pdf')

    # Plot the scores
    pred=t.model(g_test)
    df_test['pred']=tf.argmax(pred.globals, axis=1)
    predsm=tf.nn.softmax(pred.globals)
    for label in labels:
        df_test[f'score{label}']=predsm[:,label]

        ax_s[label].clear()
        myplt.labels(df_test,f'score{label}','label',fmt=fmt, ax=ax_s[label])
        ax_s[label].set_yscale('log')
    fig_s.savefig(f'{MODELSTATS}/score.pdf')

# %% Save output
analysis.roc(df_test, 'score0', f'roc_{output}')
