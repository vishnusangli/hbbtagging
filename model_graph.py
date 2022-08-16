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
import os
LOADMODEL =  None #'graphindep_allfeat_na-na--256e2-nl_40'
MODELSTATS = 'model_stats'
MODELSAVE = 'saved_models'

# %% Arguments
features= ['trk_btagIp_d0','trk_btagIp_z0SinTheta', 'trk_qOverP'
, 'trk_btagIp_z0SinTheta', 'trk_btagIp_d0Uncertainty', 'trk_btagIp_z0SinThetaUncertainty']
valid_labels = [0, 1, 2]
labels=['0', '2'] #######
ref_label = 0
output='graph'
epochs=1

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

def get_current_graph(logits, true_labels, curr_label = 0, ax = None):
    """
    Wrapper function that handles plotting of a single score distribution
    subplot in an epoch?
    """
    df = pd.DataFrame(logits, columns = [f'score{i}' for i in labels])
    df['label'] = tf.argmax(true_labels, axis = 1)
    myplt.labels(df,f'score{curr_label}','label',fmt=fmt, ax=ax)
    ax.set_title(f"model {output} label{curr_label} - {label_conv[curr_label]}")

def compare_graphs(a, b):
    for i in range(len(a)):
        if not np.all(a[i] == b[i]): return False
    return True
def save_successful(old_model, new_model, test_data):
    for graph in test_data:
        if not compare_graphs(old_model(graph), new_model(graph)):
            return False
    return True

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
    #train_data, test_data, train_label, test_label = redist_batches(train_data, test_data, train_label, test_label)
    pass
print(len(train_label), len(test_label))


# %%
if LOADMODEL != None:
    lgmodels, lnorm = graphs.load_model(filepath = f"{MODELSAVE}/{LOADMODEL}")
    loaded_model = graphs.myGraphIndep('loaded_indep', pre_models=lgmodels, pre_norm = lnorm)
    train_data = [loaded_model(graph) for graph in train_data]
    test_data  = [loaded_model(graph) for graph in test_data]
# %% Training procedure

class Trainer:
    def __init__(self, model):
        # Model to keep track of
        self.model= model

        # Training tools
        self.stat = pd.DataFrame(columns=['train_loss','test_loss', 'train_aoc_gbb', 'train_aoc_other', 'test_aoc_gbb', 'test_aoc_other'])
        self.opt  = snt.optimizers.Adam(learning_rate=0.01)
        self.method = 'global'

    def step(self, train_graphs: list, train_labels: list, test_graphs: list, test_labels: list, epoch: int):
        """Performs one optimizer step on a single mini-batch."""
        # Testing
        test_loss=0.
        train_loss = 0.
        if len(test_graphs) > 0:
            for i in range(len(test_labels)):
                pred = self.model(test_graphs[i])
                logits = self.give_preds(pred) #pred.globals
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
                logits = self.give_preds(pred) #pred.globals
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
            preds.append(self.give_preds(pred))
            true_label_list.append(tf.argmax(label_list[graph], axis = 1))
        return np.concatenate(true_label_list, axis = 0), tf.nn.softmax(np.concatenate(preds, axis = 0))
    
    def operate(self, data):
        return [self.model(gtup) for gtup in data]
    
    def give_preds(self, graph):
        return graph.globals
        if self.method =='global':
            return graph.globals
        elif self.method == 'node':
            return graph.nodes

# %%
#graph_indep = gn.modules.GraphIndependent()
#model = graphs.INModel(len(labels), nglayers=0)
#model = graphs.GNModel(nlabels=len(labels), nlayers=1, OUTPUT_NODE_SIZE=3)
#model = graphs.DSModel(OUTPUT_NODE_MLP = [3], nlabels=3, nlayers=1, hid_layers=[256, 256])
#model = graphs.GIModel(nlabels = 3, hidden_layers = 2, hidden_size = 256)
use_model = graphs.myGraphIndep
#norm = graphs.myNormModel(use_globals=True)
#model = graphs.myGraphIndep(output, nlabels = 3, node_layers = [256, 256], global_layers=None, pre_norm = None)
#t = Trainer(model)

# %%
iNMODEL = [None, None]
iGIMODEL = [None, None]
iDSMODEL = [None]
class DSModel(snt.Module):
    def __init__(self):
        super().__init__()
        temp = snt.LayerNorm(0, create_scale=True, create_offset=True)
        self.norm = gn.modules.GraphIndependent(
            global_model_fn=lambda: temp
        )
        iNMODEL.append(temp)
        
        temp = snt.nets.MLP([256, 256, 3])
        self.layers = []
        gi_network = gn.modules.GraphIndependent(
            global_model_fn=lambda: temp
        )
        iGIMODEL.append(temp)

        temp1 = snt.nets.MLP([256, 256, 7])
        temp2 = snt.nets.MLP([256, 256, 3])
        ds_network = gn.modules.DeepSets(
            node_model_fn=lambda: temp1,
            global_model_fn=lambda: temp2
        )
        iDSMODEL.append(temp1)
        iDSMODEL.append(temp2)
        self.layers.append(gi_network)
        self.layers.append(ds_network)
            
    def __call__(self, graph):
        graph = self.norm(graph)
        for layer in self.layers:
            graph = layer(graph)
        return graph
model = DSModel()
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
#model.save(path=f"{MODELSAVE}")
graphs.create_folder(f"{MODELSAVE}/{output}")
graphs.save_model(iNMODEL, f"{MODELSAVE}/{output}/norm")
graphs.save_model(iGIMODEL, f"{MODELSAVE}/{output}/graphindep")
graphs.save_model(iDSMODEL, f"{MODELSAVE}/{output}/deepset")
# %%
#gmodels, norm = graphs.load_model(filepath = f"{MODELSAVE}/{output}")
#loaded_model = use_model('new_indep', pre_models=gmodels, pre_norm = norm)
# %%
a = True #save_successful(model, loaded_model, test_data)
if a:
    print(f"Saved Successfully")
else:
    print(f"Save Unsuccessful")
# %%
