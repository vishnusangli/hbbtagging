"""
This is a specific graph net training file for creating compound models

Since graph networks return graphs themselves and have various architectures, I intend to explore the training 
of a compound model consisting of separate models invidually trained in succession

My first idea is to train a graph independent network (only global model) and use 
its output to train a deep set network
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
MODELSTATS = 'model_stats_v2'
# %% Arguments
features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP'
, 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
valid_labels = [0, 1, 2]
labels=['0', '1', '2'] #######
ref_label = 0
output='graph_comp'
epochs=10

if 'ipykernel_launcher' not in sys.argv[0]: # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Train GNN from track features')
    parser.add_argument('features', nargs='*', default=features, help='Features to train on.')
    parser.add_argument('--output', type=str, default=output, help='Output name.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train.')
    parser.add_argument('--labels', type = str, nargs='+', default = labels, help="Labels used for model")
    args = parser.parse_args()

    features = args.features
    output = args.output
    epochs = args.epochs
    labels = args.labels

labels = [int(i) for i in labels]
labels.sort()
for i in labels:
    assert i in valid_labels, f"Incorrect label provided, {i}"

strlabels=list(map(lambda l: f'label{l}', labels))
label_conv = ["hbb","QCD(bb)","QCD(other)"]
excluded_labels = list(set(valid_labels) - set(labels))
excluded_labels.sort()
# Formatting
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
label_ratio = [0.47, 0.06, 0.47]
argmax_factor = 1
if len(labels) == 2:
    import warnings
    warnings.simplefilter("ignore")
    new_labels =[0] * 3
    for i in labels:
        new_labels[i] = 0.5
        if i == 2:
            argmax_factor = 2
    label_ratio = new_labels

batch_size = 10000

print(f"Load Training")
train_loader = data.GraphLoader(signals, backs, graph_dir='feature_graphs')
train_data, train_label = data.load_all(train_loader, batch_size=batch_size, ratio=label_ratio,
                                    num_batches= 30)
print(f"Load Testing")
test_loader = data.GraphLoader(signals, backs, tag = 'r9364', graph_dir='feature_graphs')
test_data, test_label = data.load_all(test_loader, batch_size=batch_size, ratio=label_ratio,
                                    num_batches= 5)
np.sum(train_label[0], axis = 0)
# %%
test_loader, train_loader = None, None

# %%
def myargmax(x, axis = 0):
    '''
    A function decorator that crafts argmax
    in a manner that is more desireable -- representing 
    the actual labels rather than simple index-wise
    argmax
    '''
    temp = np.argmax(x, axis)
    if len(labels) == 2:
        temp = np.where(temp == 0, labels[0], temp)
        temp = np.where(temp == 1, labels[1], temp)
    return temp

def craft_preds(logits, impute = excluded_labels, impute_val = 0):
    """
    This function accomodates custom number of output logit labels while
    retaining the overall structure. \n
    It inserts dummy columns into the predictions to generate the standard
    3-label output.
    """
    if logits.shape[1] == 3: return logits
    for i in impute[::-1]:
        logits = np.insert(logits, i, impute_val, axis = 1)
    assert logits.shape[1] == 3, "Something went wrong"
    return logits
# %%
"""
Since there is less 

"""
# Redistribute batches -->

def redist_batches(train_data, test_data, train_label, test_label, train_ratio = 0.8):
    """
    When there's less data, we pull from both tags and redistribute batches, 
    so that there is enough for training. \n
    This is mainly required for label QCD (bb) data
    """
    train_data.extend(test_data)
    train_label.extend(test_label)

    num_training = int(train_ratio * len(train_label))
    num_testing = len(train_label) - num_training

    assert num_testing > 0 and num_training > 0, "Incorrect split"

    test_data, test_label = train_data[num_training:], train_label[num_training:]
    train_data, train_label = train_data[:num_training], train_label[:num_training]
    return train_data, test_data, train_label, test_label

if len(train_label) * batch_size < 100000: #Doesn't matter if redistribution helps send training size over threshold
    train_data, test_data, train_label, test_label = redist_batches(train_data, test_data, train_label, test_label)
# %%
def get_current_graph(logits, true_labels, curr_label = 0, ax = None):
    """
    Wrapper function that handles plotting of a single score distribution
    subplot in an epoch?
    """
    df = pd.DataFrame(logits, columns = [f'score{i}' for i in labels])
    df['label'] = myargmax(true_labels, axis = 1)
    myplt.labels(df,f'score{curr_label}','label',fmt=fmt, ax=ax)
    ax.set_title(f"model {output} label{curr_label} - {label_conv[curr_label]}")
# Training procedure
class Trainer:
    def __init__(self, model):
        # Model to keep track of
        self.model= model

        # Training tools
        self.stat = pd.DataFrame(columns=['train_loss','test_loss', 'train_aoc_gbb', 'train_aoc_other', 'test_aoc_gbb', 'test_aoc_other'])
        self.opt  = snt.optimizers.Adam(learning_rate=0.01)

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
                test_aoc = analysis.aoc(np.array(tf.nn.softmax(logits)), myargmax(test_labels[i], axis = 1), score = ref_label)

                fig_s,ax_s=plt.subplots(ncols=len(labels),figsize=(24,8)) #scores

                logits = tf.nn.softmax(logits, axis = 1)
                [get_current_graph(logits, test_labels[i], labels[lab], ax_s[lab]) for lab in range(len(labels))]
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
                train_aoc = analysis.aoc(np.array(tf.nn.softmax(logits)), myargmax(train_labels[i], axis = 1), score = ref_label)

            train_loss += loss


        test_loss = np.divide(test_loss, len(test_labels))
        train_loss = np.divide(train_loss, len(train_labels))
        # save training status
        """        
        for label in excluded_labels:
            for i in [train_aoc, test_aoc]:
                i.insert(label, np.nan)"""

        epoch_stats = {'train_loss':float(loss), 'test_loss':float(test_loss), 'train_aoc_gbb':float(train_aoc[1]), 
                                    'train_aoc_other':float(train_aoc[2]), 'test_aoc_gbb':float(test_aoc[1]), 
                                    'test_aoc_other':float(test_aoc[2])}
        self.stat = self.stat.append(epoch_stats, ignore_index = True)
        return train_loss

    def test_model_preload(self, graph_list, label_list):
        preds = []
        true_label_list = []
        for graph in range(len(graph_list)):
            pred = self.model(graph_list[graph])
            preds.append(pred.globals)
            true_label_list.append(myargmax(label_list[graph], axis = 1))
        return np.concatenate(true_label_list, axis = 0), tf.nn.softmax(np.concatenate(preds, axis = 0))
# Training
def train_model(t: Trainer, output: str, epochs: int, suffix: str = 'primary'):
    """
    This function contains the model training so 
    """

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
        fig_t.savefig(f'{MODELSTATS}/{suffix}/training.png')
        ax_t.set_ylim(1e-1, 1e1)
        fig_t.savefig(f'{MODELSTATS}/{suffix}/training.pdf')
        #plt.show()
        plt.close(fig_t)
        
        true_labels, predsm = t.test_model_preload(test_data, test_label)
        analysis.bare_roc(np.array(predsm), true_labels, ref_label, f'roc_{output}-{suffix}', epoch_roc=True, epoch_num = epoch, file_dir = {MODELSTATS}/{suffix})

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
        plt.savefig(f'{MODELSTATS}/{suffix}/train_aoc.pdf')
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
        plt.savefig(f'{MODELSTATS}/{suffix}/test_aoc.pdf')
        plt.close()
        plt.clf()
# %%
model_list = []
#model = graphs.INModel(len(labels), nglayers=0)
#model = graphs.GNModel(nlabels=len(labels), nlayers=1, OUTPUT_NODE_SIZE=3)
#model = graphs.DSModel(OUTPUT_NODE_SIZE=3, nlabels=3, nlayers=1)
pmodel = graphs.GIModel(nlabels = len(labels), hidden_layers = 2, hidden_size = 256)
t = Trainer(pmodel)
# %%
# PRIMARY MODEL TRAINING
train_model(t, output, epochs = epochs, suffix = 'primary')
model_list.append(pmodel)
# %%
# REWRITING DATASET W.R.T MODEL
train_data = [t.model(gtup) for gtup in train_data]
test_data = [t.model(gtup) for gtup in test_data]

# %%
#model = graphs.INModel(len(labels), nglayers=0)
#model = graphs.GNModel(nlabels=len(labels), nlayers=1, OUTPUT_NODE_SIZE=3)
smodel = graphs.DSModel(OUTPUT_NODE_MLP=[3], nlabels=3, hid_layers =[256, 256])
#primary_model = graphs.GIModel(nlabels = len(labels), hidden_layers = 2, hidden_size = 256)
t = Trainer(smodel)
# %%
# SECONDARY MODEL TRAINING
train_model(t, output, epochs = epochs, suffix = 'secondary')
model_list.append(smodel)
# %% Save output
true_labels, predsm = t.test_model_preload(test_data, test_label)
analysis.bare_roc(np.array(predsm), true_labels, ref_label, f'roc_{output}')

# %%
def mult_model_output(model_list, graph_list, label_list):
    """
    Redundanf function, Use when loading?
    """
    preds = []
    true_label_list = []
    for graph_index in range(len(graph_list)):
        pred = graph_list[graph_index]
        for model in model_list:
            pred = model(graph_list[graph_index])
        preds.append(pred.globals)
        true_label_list.append(myargmax(label_list[graph_index], axis = 1))
    return np.concatenate(true_label_list, axis = 0), tf.nn.softmax(np.concatenate(preds, axis = 0))