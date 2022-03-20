# %%
#%load_ext autoreload
#%autoreload 2

#%%
import sys
from unicodedata import name
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

from hbbgbb.models import SimpleCNN
import explore_files
from tqdm import tqdm
STATSDIR = 'data_stats'
MODELSTATS = 'model_stats'

# %% Arguments
output='cnn'
epochs= 10
if 'ipykernel_launcher' not in sys.argv[0]: # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Train a CNN with calorimeter images')
    parser.add_argument('--output', type=str, default=output, help='Output name.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train.')
    args = parser.parse_args()

    output = args.output
    epochs = args.epochs

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %%
train_data, train_label = explore_files.load_calo_data()
test_data, test_label = explore_files.load_calo_data('r9364')

# %%
# Feature Engineering

train_data = explore_files.Feature_Eng.current(train_data)
explore_files.avg_img_perlabel(train_data, train_label, name = "avg_img_train")

test_data = explore_files.Feature_Eng.current(test_data)
explore_files.avg_img_perlabel(test_data, test_label, name = "avg_img_test")

# %%
class CNN_BatchTrainer:
    def __init__(self, model = SimpleCNN.CNNModel):
        self.model = model()

        self.loss = []
        self.stat = pd.DataFrame(columns=['train_loss','test_loss'])
        self.opt = snt.optimizers.SGD(learning_rate=0.1)
        self.loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
    def Batch_Trainer(self, data, label, start, end):
        batch_data, batch_label = data[start: end], label[start: end]
        pred = self.model(batch_data, is_training = True)
        with tf.GradientTape() as tape:
            
            loss = self.loss_fn(batch_label, pred)
        return loss

    
    def master_train(self, data, label, epochs, batch_size = 16):

        for epoch in range(epochs):
            for batch in tqdm(range((data.shape[0]//batch_size) + 1)):
                start, end = (batch * batch_size), min(batch_size * (batch + 1), data.shape[0])
                batch_loss = self.Batch_Trainer(data, label, start, end)



# %%
shape = (explore_files.IMG_SIZE, explore_files.IMG_SIZE)

train_data, train_label = tf.convert_to_tensor(train_data, dtype = tf.float32), tf.convert_to_tensor(train_label, dtype = tf.float32)
test_data, test_label = tf.convert_to_tensor(test_data, dtype = tf.float32), tf.convert_to_tensor(test_label, dtype = tf.float32)

train_data = tf.reshape(train_data, (-1, *shape, 1))
test_data = tf.reshape(test_data, (-1, *shape, 1))
train_data.shape
# %%
t = CNN_BatchTrainer()
# %%
fig_s,ax_s=plt.subplots(ncols=3,figsize=(24,8))
fig_t,ax_t=plt.subplots(figsize=(8,8))

batch_size = 32
epoch_stat=pd.DataFrame(columns=['train_loss', 'test_loss'])
for epoch in tqdm(range(epochs)):
    for batch in range((train_data.shape[0]//batch_size) + 1):
        start, end = (batch * batch_size), min(batch_size * (batch + 1), train_data.shape[0])
        batch_loss = t.Batch_Trainer(train_data, train_label, start, end)
    with tf.GradientTape() as tape:
        pred = t.model(train_data)
        loss = t.loss_fn(train_label, pred)

        test_pred = t.model(test_data)
        test_loss = t.loss_fn(test_label, test_pred)
        params = t.model.trainable_variables
        grads = tape.gradient(loss, params)

        t.opt.apply(grads, params)
        loss = float(tf.reduce_mean(loss))
        test_loss = float(tf.reduce_mean(test_loss))
        epoch_stat = epoch_stat.append({'train_loss':float(loss), 'test_loss':float(test_loss)}, ignore_index=True)
    

    # Plot the status of the training
    ax_t.clear()
    ax_t.plot(epoch_stat.train_loss,label='Training')
    ax_t.plot(epoch_stat.test_loss ,label='Test')
    ax_t.set_yscale('log')
    ax_t.set_ylabel('loss')
    ax_t.set_ylim(1e-1, 1e3)
    ax_t.set_xlabel('epoch')
    ax_t.legend()
    fig_t.savefig(f'{MODELSTATS}/training.pdf')

    # Plot the scores

    #df_test['pred']=tf.argmax(pred.globals, axis=1)
for tensor in t.model.trainable_variables:
    print("{} : {}".format(tensor.name, tensor.shape))

pred=t.model(test_data)
df = pd.DataFrame(tf.argmax(pred, axis=1), columns = ['pred'])
predsm=tf.nn.softmax(pred)
df['score0']=predsm[:,0]
df['score1']=predsm[:,1]
df['score2']=predsm[:,2]

df['label'] = test_label[:, 1] + (2* test_label[:, 2])
# %%
myplt.labels(df,'score0','label',fmt=fmt)
plt.title(f"model {output} label0 - hbb")
plt.savefig(f'{MODELSTATS}/score0.pdf')
plt.show()
plt.clf()
# %%
myplt.labels(df,'score1','label',fmt=fmt)
plt.title(f"model {output} label1 - QCD(bb)")
plt.savefig(f'{MODELSTATS}/score1.pdf')
plt.show()
plt.clf()
# %%
myplt.labels(df,'score2','label',fmt=fmt)
plt.title(f"model {output} label2 - QCD(other)")
plt.savefig(f'{MODELSTATS}/score2.pdf')
plt.show()
plt.clf()
# %%
analysis.roc(df, 'score0', f'roc_{output}')
