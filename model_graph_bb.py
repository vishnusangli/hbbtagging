"""
This is a specific graph net training file for optimizing QCD (bb) training.

"""
# %%
import sys

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
MODELSTATS = 'model_stats'
# %% Arguments
features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP'
, 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
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
label_conv = ["hbb","QCD(bb)","QCD(other)"]
# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %% Load per jet information
trk_features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 
'trk_qOverP', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
calo_features = ['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
signals = ["1100", "1200", "1400"]
backs = ["5"]

# %% Load Data
"""
Sample ratios
[0.47, 0.06, 0.47]
[0.4999, 0.0002, 0.4999]
[0.4999, 0.4999, 0.0002]

"""
print(f"Load Training")
train_loader = data.GraphLoader(signals, backs, graph_dir='feature_graphs')
train_data, train_label = data.load_all(train_loader, batch_size=5000, ratio=[0.4999, 0.4999, 0.0002],
                                    num_batches= 100)
print(f"Load Testing")
test_loader = data.GraphLoader(signals, backs, tag = 'r9364', graph_dir='feature_graphs')
test_data, test_label = data.load_all(test_loader, batch_size=5000, ratio=[0.4999, 0.4999, 0.0002],
                                    num_batches= 10)
# %%
test_loader, train_loader = None, None

# %%
# Redistribute batches -->

# %%
def get_current_graph(logits, true_labels, curr_label = 0, ax = None):
    """
    Wrapper function that handles plotting of a single score distribution
    subplot in an epoch?
    """
    df = pd.DataFrame(logits, columns = [f'score{i}' for i in labels])
    df['label'] = tf.argmax(true_labels, axis = 1)
    myplt.labels(df,f'score{curr_label}','label',fmt=fmt, ax=ax)
    ax.set_title(f"model {output} label{curr_label} - {label_conv[curr_label]}")
    '''
    plt.title(f"model {output} label{curr_label} - {label_conv[curr_label]}")
    plt.savefig(f'{MODELSTATS}/score{curr_label}.pdf')
    plt.show()
    plt.clf()
    '''
# %% Training procedure
class Trainer:
    def __init__(self, model):
        # Model to keep track of
        self.model= model

        # Training tools
        self.stat = pd.DataFrame(columns=['train_loss','test_loss', 'train_aoc_gbb', 'train_aoc_other', 'test_aoc_gbb', 'test_aoc_other'])
        self.opt  = snt.optimizers.Adam(learning_rate=0.1)

    def step(self, train_graphs: list, train_labels: list, test_graphs: list, test_labels: list, epoch: int):
        """Performs one optimizer step on a single mini-batch."""
        # Testing
        test_loss=0.
        train_loss = 0.
        if len(test_graphs) > 0:
            for i in range(len(test_labels)):
                pred = self.model(test_graphs[i])
                logits = pred.globals
                loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                                labels = test_labels[i])
                test_loss += tf.reduce_mean(loss)
                test_aoc = analysis.aoc(np.array(tf.nn.softmax(logits)), tf.argmax(test_labels[i], axis = 1), score = 0)

                fig_s,ax_s=plt.subplots(ncols=len(labels),figsize=(24,8)) #scores

                logits = tf.nn.softmax(logits, axis = 1)
                [get_current_graph(logits, test_labels[i], lab, ax_s[lab]) for lab in range(len(labels))]
                plt.suptitle(f"model {output}: epoch-{epoch}")
                fig_s.savefig(f'scores/score-{epoch}.pdf')
                plt.close(fig_s)

        # Training
        for i in range(len(train_labels)):
            with tf.GradientTape() as tape:
                pred = self.model(train_graphs[i])
                logits = pred.globals
                loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                                labels = train_labels[i])
                loss =  tf.reduce_mean(loss)

                params = self.model.trainable_variables
                grads = tape.gradient(loss, params)
                self.opt.apply(grads, params)
                train_aoc = analysis.aoc(np.array(tf.nn.softmax(logits)), tf.argmax(train_labels[i], axis = 1), score = 0)

            train_loss += loss


        test_loss = np.divide(test_loss, len(test_labels))
        train_loss = np.divide(train_loss, len(train_labels))
        # save training status
        self.stat=self.stat.append({'train_loss':float(loss), 'test_loss':float(test_loss), 'train_aoc_gbb':float(train_aoc[1]), 
                                    'train_aoc_other':float(train_aoc[2]), 'test_aoc_gbb':float(test_aoc[1]), 
                                    'test_aoc_other':float(test_aoc[2])}, ignore_index=True)
        return train_loss
    
    def test_model_preload(self, graph_list, label_list):
        preds = []
        true_label_list = []
        for graph in range(len(graph_list)):
            pred = self.model(graph_list[graph])
            preds.append(pred.globals)
            true_label_list.append(tf.argmax(label_list[graph], axis = 1))
        return np.concatenate(true_label_list, axis = 0), tf.nn.softmax(np.concatenate(preds, axis = 0))
# %%
#graph_indep = gn.modules.GraphIndependent()
#model = graphs.INModel(len(labels), nglayers=0)
#model = graphs.GNModel(nlabels=len(labels), nlayers=1, OUTPUT_NODE_SIZE=3)
#model = graphs.DSModel(OUTPUT_NODE_SIZE=3, nlabels=3, nlayers=1)
model = graphs.GIModel(nlabels = 3, hidden_layers = 2, hidden_size = 256)
t = Trainer(model)

# %% Training

for epoch in tqdm.trange(epochs):
    loss=float(t.step(train_data, train_label, test_data, test_label, epoch))

    # Plot the status of the training
    fig_t,ax_t=plt.subplots(figsize=(8,8), facecolor = 'white') #training
    ax_t.clear()
    ax_t.set_title(f"{output} Training curve")
    ax_t.plot(t.stat.train_loss,label='Training')
    ax_t.plot(t.stat.test_loss ,label='Test')
    ax_t.set_yscale('log')
    ax_t.set_ylabel('loss')
    ax_t.set_xlabel('epoch')
    ax_t.grid()
    ax_t.legend()
    fig_t.savefig(f'{settings.modelstats}/training.png')
    ax_t.set_ylim(1e-2, 1e1)
    fig_t.savefig(f'{settings.modelstats}/training.pdf')
    #plt.show()
    plt.close(fig_t)
    
    true_labels, predsm = t.test_model_preload(test_data, test_label)
    analysis.bare_roc(np.array(predsm), true_labels, 0, f'roc_{output}', epoch_roc=True, epoch_num = epoch)

    # Plot the scores

    # %% Plotting train aoc
    plt.figure(figsize=(8,8))
    plt.plot(t.stat.train_aoc_gbb, label = "QCD(gbb)")
    plt.plot(t.stat.train_aoc_other, label = "QCD(other)")
    plt.title(f"{output} ROC training AOC curve")
    plt.ylabel('AOC')
    plt.yscale("log")
    plt.ylim(1e-3, 1)
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.savefig(f'{MODELSTATS}/train_aoc.pdf')
    #plt.show()
    plt.close()
    plt.clf()

    # %% Plotting test aoc
    plt.figure(figsize=(8,8))
    plt.plot(t.stat.test_aoc_gbb, label = "QCD(gbb)")
    plt.plot(t.stat.test_aoc_other, label = "QCD(other)")
    plt.title(f"{output} ROC testing AOC curve")
    plt.ylabel('AOC')
    plt.yscale("log")
    plt.ylim(1e-3, 1)
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.savefig(f'{MODELSTATS}/test_aoc.pdf')
    #plt.show()
    plt.close()
    plt.clf()
# %% Save output
true_labels, predsm = t.test_model_preload(test_data, test_label)
analysis.bare_roc(np.array(predsm), true_labels, 0, f'roc_{output}')
# %%
