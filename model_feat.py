# %%
#%load_ext autoreload
#%autoreload 2

#%%
import sys

import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

from hbbgbb.models import SimpleModel
from tqdm import tqdm
STATSDIR = 'data_stats'
MODELSTATS = 'model_stats'
MODELDIR = 'saved_models'
# %% Arguments
features=['mass', 'C2','D2','e3','Tau32_wta','Split12','Split23']
output='feat'
epochs=10
if 'ipykernel_launcher' not in sys.argv[0]: # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Train NN from features')
    parser.add_argument('features', nargs='*', default=features, help='Features to train on.')
    parser.add_argument('--output', type=str, default=output, help='Output name.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train.')
    args = parser.parse_args()

    features = args.features
    output = args.output
    epochs = args.epochs

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %% Load the datset
signals = ["1200", "1400"]
backs = ["6"]
labels = [0, 1, 2]
strlabels=list(map(lambda l: f'label{l}', labels))
 # %%
signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(sig_mass = signals, bag_jx=backs, tag = 'r10201')
master_arr = signal_arrs + backg_arrs
[data.label(i) for i in master_arr]
#Remove label 3/only required data
master_arr = [i[np.any(i[strlabels], axis = 1)].copy() for i in master_arr]
# Combine and shuffle dataset into one
master_data = [np.array(i[features]) for i in master_arr]
master_label = [np.array(i[strlabels]) for i in master_arr]
master_data, master_label = data.merge_shuffle(master_label, master_data)


test_true_labels = tf.argmax(master_label, axis = 1) #used for roc curve
# %% Create tensors of features
test_feat=tf.convert_to_tensor(master_data)
test_label=tf.convert_to_tensor(test_true_labels)

# %%
# Training Dataset
signal_arrs, backg_arrs, new_sig_mass, new_bag_jx = data.load_newdata(sig_mass = signals, bag_jx=backs, tag = 'r9364')
master_arr = signal_arrs + backg_arrs
[data.label(i) for i in master_arr]
#Remove label 3/only required data
master_arr = [i[np.any(i[strlabels], axis = 1)].copy() for i in master_arr]
# Combine and shuffle dataset into one
master_data = [np.array(i[features]) for i in master_arr]
master_label = [np.array(i[strlabels]) for i in master_arr]
master_data, master_label = data.merge_shuffle(master_label, master_data)
# %%
train_true_labels = tf.argmax(master_label, axis = 1) #used for roc curve
feat=tf.convert_to_tensor(master_data)
labels=tf.convert_to_tensor(train_true_labels)

# %%
mlp=SimpleModel.SimpleModel()

# %%
opt = snt.optimizers.SGD(learning_rate=0.1)

def step(feat,labels, label_col):
  """Performs one optimizer step on a single mini-batch."""
  with tf.GradientTape() as tape:
    logits = mlp(feat, is_training=True)
    aoc = analysis.aoc(np.array(tf.nn.softmax(logits)), pd.Series(label_col), score = 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    loss = tf.reduce_mean(loss)


  params = mlp.trainable_variables
  grads = tape.gradient(loss, params)
  opt.apply(grads, params)
  return loss, aoc[1:]

def gen_testloss(feat, labels, label_col):
  """
  Generates a step with loss without training
  """
  with tf.GradientTape() as tape:
    logits = mlp(feat)
    aoc = analysis.aoc(np.array(tf.nn.softmax(logits)), pd.Series(label_col), score = 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    loss = tf.reduce_mean(loss)
  return loss, aoc[1:]

# %% Training
df_stat=pd.DataFrame(columns=['epoch','loss', 'test_loss', 'aoc_gbb', 'aoc_other', 'test_aoc_gbb', 'test_aoc_other'])
for epoch in tqdm(range(epochs)):
    loss, aoc =step(feat,labels, train_true_labels)
  
    test_loss, test_aoc = gen_testloss(test_feat, test_label, test_true_labels)

    df_stat=df_stat.append({'epoch':epoch,'loss':float(loss), 'test_loss':float(test_loss), 'aoc_gbb':float(aoc[0]), 'aoc_other':float(aoc[1]), 'test_aoc_gbb':float(test_aoc[0]), 'test_aoc_other':float(test_aoc[1])}, ignore_index=True)

# %% Plotting loss
plt.plot(df_stat.epoch, df_stat.loss, label = "training")
plt.plot(df_stat.epoch, df_stat.test_loss, label = "test")
plt.title(f"{output} Training curve")
plt.yscale('log')
plt.ylabel('loss')
plt.ylim(1e-1, 1e1)
plt.xlabel('epoch')
plt.legend()
plt.savefig(f'{MODELSTATS}/training.pdf')
plt.show()
plt.clf()

# %% Plotting train aoc
plt.plot(df_stat.epoch, df_stat.aoc_gbb, label = "QCD(gbb)")
plt.plot(df_stat.epoch, df_stat.aoc_other, label = "QCD(other)")
plt.title(f"{output} ROC training AOC curve")

plt.ylabel('AOC')
plt.ylim(0, 1)

plt.xlabel('epoch')
plt.legend()
plt.savefig(f'{MODELSTATS}/train_aoc.pdf')
plt.show()
plt.clf()

# %% Plotting test aoc
plt.plot(df_stat.epoch, df_stat.test_aoc_gbb, label = "QCD(gbb)")
plt.plot(df_stat.epoch, df_stat.test_aoc_other, label = "QCD(other)")
plt.title(f"{output} ROC testing AOC curve")

plt.ylabel('AOC')
plt.ylim(0, 1)

plt.xlabel('epoch')
plt.legend()
plt.savefig(f'{MODELSTATS}/test_aoc.pdf')
plt.show()
plt.clf()
# %% Generating the predictions
pred=mlp(feat)
predsm=tf.nn.softmax(pred)
data.write(f"{MODELDIR}/{output}-train", predsm)
del(predsm)
del(pred)

pred=mlp(test_feat)

predsm=tf.nn.softmax(pred)
data.write(f"{MODELDIR}/{output}-test", predsm)


# %% Plot distributions of the two predictions


# %% Calculate ROC curves
analysis.bare_roc(np.array(predsm), train_true_labels, 0, output)