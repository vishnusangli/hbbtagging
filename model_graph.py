# %%
#%load_ext autoreload
#%autoreload 2

# %%
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
LOADMODEL = "simplenn"
# %% Arguments
features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
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
trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
signals = ["1100", "1200", "1400"]
backs = ["5"]


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

    def step(self, train_loader: data.GraphLoader, test_loader: data.GraphLoader):
        """Performs one optimizer step on a single mini-batch."""
        # Write test data
        test_loss=0.
        if test_loader is not None:
            while not test_loader.is_finished():
                batch_g, batch_l = test_loader.give_batch(label_ratio = [0.495, 0.1, 0.495], batch_size=10000)
                pred = self.model(batch_g)
            logits = pred.globals
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=batch_l)
            test_loss = tf.reduce_mean(loss)

        # Training
        with tf.GradientTape() as tape:
            while not train_loader.is_finished():
                batch_g, batch_l = test_loader.give_batch(label_ratio = [0.495, 0.1, 0.495], batch_size=10000)
                pred = self.model(batch_g)

            logits=pred.globals
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=batch_l)
            loss = tf.reduce_mean(loss)

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.opt.apply(grads, params)

        # save training status
        self.stat=self.stat.append({'train_loss':float(loss), 'test_loss':float(test_loss)}, ignore_index=True)

        return loss
    
    def test_model(self, test_loader):
        
        true_labels = []
        preds = []
        for i in range(3):
            print(i)
            batch_g, batch_l = test_loader.give_batch(label_ratio = [0.495, 0.1, 0.495], batch_size=10000)
            pred = self.model(batch_g)
            preds.append(tf.nn.softmax(pred.globals))
            true_labels.append(batch_l)
        return np.concatenate(true_labels, axis = 0), np.concatenate(preds, axis = 0)

        while not test_loader.is_finished():
            batch_g, batch_l = test_loader.give_batch(label_ratio = [0.495, 0.1, 0.495], batch_size=10000)
            pred = self.model(batch_g)
            preds.append(tf.nn.softmax(pred.globals))
            true_labels.append(batch_l)

        return np.concatenate(true_labels, axis = 0), np.concatenate(preds, axis = 0)

# %% Prepare for training
model = graphs.INModel(len(labels),2)
t = Trainer(model)

# %% Training
fig_s,ax_s=plt.subplots(ncols=3,figsize=(24,8))
fig_t,ax_t=plt.subplots(figsize=(8,8))

for epoch in tqdm.trange(epochs):
    train_loader = data.GraphLoader(signals, backs)
    test_loader = data.GraphLoader(signals, backs, tag = 'r9364')

    loss=float(t.step(train_loader, test_loader))

    # Plot the status of the training
    ax_t.clear()
    ax_t.plot(t.stat.train_loss,label='Training')
    ax_t.plot(t.stat.test_loss ,label='Test')
    ax_t.set_yscale('log')
    ax_t.set_ylabel('loss')
    ax_t.set_ylim(1e-1, 1e3)
    ax_t.set_xlabel('epoch')
    ax_t.legend()
    fig_t.savefig(f'{settings.modelstats}/training.pdf')

    # Plot the scores
true_labels, predsm = t.test_model(data.GraphLoader(signals, backs))

# %% Save output
analysis.bare_roc(np.array(predsm), true_labels, 0, f'roc_{output}')
