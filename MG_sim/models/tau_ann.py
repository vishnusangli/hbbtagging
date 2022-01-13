# %% Imports 

import sys
sys.path.insert(0, '/global/u1/v/vsangli/starters/hbbtagging/')

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, backend
import matplotlib.pyplot as plt
from MG_sim.file_read.file_support import *
from common.model_support import *
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
misc_vals, parent_data, master_label, jet_codes = get_files(BGROUND_Label, M_SIG_Label, I_SIG_Label, misc_features=8, seed = 0)
count_labels(master_label)

# %%
#Feature Engineering
print(f"NaNs : {np.sum(np.isnan(misc_vals))}")
class myVar_FeatureEngineering:
    def try_1(data):
        sep = np.log2((data * 10) + 0.5)
        for i in range(sep.shape[1]):
            curr_min = np.nanmin(sep[:, i])
            sep[:, i] = custom_norm(sep[:, i], curr_min, np.nanmax(sep[:, i]) - curr_min)
        return sep
    def try_2(vals):
        output = vals * 1
        for i in range(0, vals.shape[1]):
            data = vals[:, i]
            sep = output[:, i]
            if i == 5:
                pass
            else:
                sep = np.log2((sep * 10) + 0.5)
                for i in range(sep.shape[1]):
                    curr_min = np.nanmin(sep[:, i])
                    sep[:, i] = custom_norm(sep[:, i], curr_min, np.nanmax(sep[:, i]) - curr_min)
        return sep
    current = try_1
selected_data = [0, 1, 2, 3, 4, 7]
print([FEATURE_VARS[i] for i in selected_data])
master_data = myVar_FeatureEngineering.current(misc_vals[:, selected_data])
# %%
#Plot Data Distributions
fig, ax = plt.subplots(3, 2, figsize = (10, 10), sharex = False)
plt.suptitle("Distributions")
for i in range(0, 6):
    plt.subplot(3, 2, i + 1)
    label_0, label_1, label_2 = sort_disc_vals(master_data[:, i], master_label)
    n, hist_0, hist_1, hist_2 =  generate_labelhist(label_0, label_1, label_2, master_data[:, i])
    plt_hist(n, hist_0, hist_1, hist_2, f"{FEATURE_VARS[selected_data[i]]}")

# %%
    
# %% Basic model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=( len(selected_data))),
    tf.keras.layers.Dense(7, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', "sparse_categorical_accuracy"])
# %% 
# Train/test split data
train_data, train_label, train_code, test_data, test_label, test_code = ter_split_data(master_data, master_label, jet_codes, train_ratio=0.8)
count_labels(train_label), count_labels(test_label)
# %%
# fit model
NUM_EPOCHS = 20
BATCH_SIZE = 30
lim = len(train_data)
first = train_data[:lim]
model.fit(x = first, y = train_label[0:lim], validation_data = (test_data[0:lim], test_label[0:lim]), batch_size = BATCH_SIZE, epochs=NUM_EPOCHS)
# %%
plot_accuracy_loss(model, NUM_EPOCHS)
# %%
# Predict on test data
num_do = test_label.shape[0]
#model_predictions = np.zeros(shape = (num_do, 3))
model_predictions = model.predict(test_data[:num_do]) 

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
# Converting to DataFrame for easy access
#Only for user study, dataframe not needed for functions
lims = [0, len(model_predictions)]
first = pd.DataFrame(test_data[lims[0]:lims[1]], columns = [FEATURE_VARS[i] for i in selected_data])
first.insert(0, "label" ,test_label[lims[0]:lims[1]])
sec = pd.DataFrame(model_predictions[lims[0]:lims[1]], columns = ["label 1", "label 2", "label 3"])
sec.insert(3, "disc_val", disc_vals)
first = first.join(sec, how = 'outer')
# %%
# Discriminating Variable Histograms
label_0, label_1, label_2 = sort_disc_vals(disc_vals, test_label)
n, hist_0, hist_1, hist_2 =  generate_labelhist(label_0, label_1, label_2, disc_vals)
plt_hist(n, hist_0, hist_1, hist_2, f"Discriminating variable for {MODELTYPE}")
# %%
#Obtain Rejection rates
plot_rej_rates(label_0, label_1, label_2, MODELTYPE) 
# %%
temp = time.localtime()
tf.keras.models.save_model( model, f"/global/homes/v/vsangli/starters/models/{MODELTYPE}/{temp.tm_mon}-{temp.tm_mday}", overwrite=True,)
# %%
############## End of Regular File ################
# %%
"""
#Load Previous Saved model
load_model_name = "12-3"
model = tf.keras.models.load_model(f"/global/homes/v/vsangli/starters/models/{MODELTYPE}/{load_model_name}")

a = pd.DataFrame(misc_vals, columns = FEATURE_VARS)
a.insert(0, "labels", master_label)
four = a[(a.Mass > 100) & (a.Mass < 150)]
sample_data = four.loc[:, ["tau1", "tau2", "tau3", "tau4", "tau5", "Mass"]]
# %%
"""
# %%
