# %% Imports 
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
from file_support import *
from model_support import *
import math
import time
import pandas as pd
# %%
#Directory and label file selection
BGROUND_Label = {0:[1]}
M_SIG_Label = {0:[0]}
I_SIG_Label = {0:[2]}
MODELTYPE = 'tau-ann'
# Retrieve Datasets
dataset_arr = []
code_arr = []
for ev_type in [BGROUND_Label, M_SIG_Label, I_SIG_Label]:
    if ev_type == BGROUND_Label:
        ev_dir = BGROUND
    elif ev_type == M_SIG_Label:
        ev_dir = M_SIG
    else:
        ev_dir = I_SIG
    for event in ev_type.keys():
        for label in ev_type[event]:
            print(ev_dir[event], label)
            curr_arr = pd.read_parquet(f"{DATA_DIR}/{ev_dir[event]}/misc_features_{label}.parquet", engine="pyarrow")
            curr_code = np.array(curr_arr.pop('code'))
            dataset_arr.append(np.array(curr_arr))
            code_arr.append(curr_code)


success, parent_data, code_label = shuffle_arrays(dataset_arr, code_arr, shape = [dataset_arr[0].shape[1]])
master_label, master_data = parent_data[:, -1], parent_data[:, :-1]
count_labels(master_label)
# %% Basic model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(7, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', "sparse_categorical_accuracy"])
# %% 
# Train/test split data
train_data, train_label, test_data, test_label = split_data(master_data, master_label, train_ratio=0.8)
count_labels(train_label), count_labels(test_label)
# %%
# %%
# fit model
NUM_EPOCHS = 20
BATCH_SIZE = 10
lim = len(train_data)
first = train_data[:lim]
model.fit(x = first, y = train_label[0:lim], validation_data = (test_data[0:lim], test_label[0:lim]), batch_size = BATCH_SIZE, epochs=NUM_EPOCHS)
# %%
plot_accuracy_loss(model, NUM_EPOCHS)
# %%
# Predict on test data
num_do = test_label.shape[0]
model_predictions = np.zeros(shape = (test_label.shape[0], 3))
for i in range(0, num_do):
    model_predictions[i] = model.predict(np.reshape(test_data[i], (1, 7)) ) 
    if i % 500 == 0:
        print(f"{i} Done")

# %%
# Generate labels from predictions
# TODO Use f1 metric -- discriminating variable
def disc_var(prediction, f1):
    """
    Discriminating variable for determining predicted variable
    """
    p0 = prediction[0]
    p1 = prediction[1]
    p2 = prediction[2]
    value = np.log(p0/(f1 * p1 + (1 - f1) * p2))
    return value

naive_pred = lambda x: np.array([np.argmax(i) for i in x])
disc_vals = np.array([disc_var(model_predictions[i], 0.9) for i in range(0, model_predictions.shape[0])])

# %%
n, hist_0, hist_1, hist_2 =  generate_labelhist(disc_vals, test_label)
plt_hist(n, hist_0, hist_1, hist_2)
# %%
# %%
temp = time.localtime()
tf.keras.models.save_model( model, f"../models/{MODELTYPE}/{temp.tm_mon}-{temp.tm_mday}", overwrite=True,)
# %%
