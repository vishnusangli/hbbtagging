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
df=data.load_data()
data.label(df)

df_test=data.load_data('r9364')
data.label(df_test)


# %% Create tensors of features
feat=tf.convert_to_tensor(df[features])
labels=tf.convert_to_tensor(df.label)

test_feat=tf.convert_to_tensor(df_test[features])
test_label=tf.convert_to_tensor(df_test.label)

# %% Create features
for feature in features+['nConstituents']:
  myplt.labels(df, feature, 'label', fmt=fmt)
  plt.savefig(f'{STATSDIR}/labels_{feature}.pdf')
  plt.show()
  plt.clf()

# %%
mlp=SimpleModel.SimpleModel()

# %%
opt = snt.optimizers.SGD(learning_rate=0.1)

def step(feat,labels, label_col):
  """Performs one optimizer step on a single mini-batch."""
  with tf.GradientTape() as tape:
    logits = mlp(feat, is_training=True)
    aoc = analysis.better_aoc(np.array(tf.nn.softmax(logits)), label_col, score = 0)
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
    aoc = analysis.better_aoc(np.array(tf.nn.softmax(logits)), label_col, score = 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    loss = tf.reduce_mean(loss)
  return loss, aoc[1:]

# %% Training
df_stat=pd.DataFrame(columns=['epoch','loss', 'test_loss', 'aoc_gbb', 'aoc_other', 'test_aoc_gbb', 'test_aoc_other'])
for epoch in tqdm(range(epochs)):
    loss, aoc =step(feat,labels, df.label)
  
    test_loss, test_aoc = gen_testloss(test_feat, test_label, df_test.label)

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
df['pred']=tf.argmax(pred, axis=1)
predsm=tf.nn.softmax(pred)
data.write(f"{MODELDIR}/{output}-train", predsm)
del(predsm)
del(pred)

pred=mlp(test_feat)
df_test['pred']=tf.argmax(pred, axis=1)
predsm=tf.nn.softmax(pred)
data.write(f"{MODELDIR}/{output}-test", predsm)

df_test['score0']=predsm[:,0]
df_test['score1']=predsm[:,1]
df_test['score2']=predsm[:,2]
# %% Plot distributions of the two predictions
for feature in features+['nConstituents']:
  myplt.labels(df_test, feature, 'label', 'pred', fmt=fmt)
  plt.savefig(f'{STATSDIR}/predictions_{feature}.pdf')
  plt.show()
  plt.clf()

# %%
myplt.labels(df_test,'score0','label',fmt=fmt)
plt.savefig(f'{MODELSTATS}/score0.pdf')
plt.title(f"model {output} label0 - hbb")
plt.show()
plt.clf()
# %%
myplt.labels(df_test,'score1','label',fmt=fmt)
plt.savefig(f'{MODELSTATS}/score1.pdf')
plt.title(f"model {output} label1 - QCD(bb)")
plt.show()
plt.clf()
# %%
myplt.labels(df_test,'score2','label',fmt=fmt)
plt.savefig(f'{MODELSTATS}/score2.pdf')
plt.title(f"model {output} label2 - QCD(other)")
plt.show()
plt.clf()

# %% Calculate ROC curves
analysis.roc(df_test, 'score0', f'roc_{output}')

# %% Save model

  # %%
analysis.aoc(df_test, 'score0')
# %%

# %%
