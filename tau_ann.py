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
misc_vals, parent_data, master_label, jet_codes = get_files(BGROUND_Label, M_SIG_Label, I_SIG_Label, misc_features=7, seed = 0)
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
    current = try_1

master_data = myVar_FeatureEngineering.current(misc_vals[:, :-1])
# %%
#Plot Data Distributions
fig, ax = plt.subplots(3, 2, figsize = (10, 10), sharex = False)
plt.suptitle("Distributions")
for i in range(0, 6):
    plt.subplot(3, 2, i + 1)
    label_0, label_1, label_2 = sort_disc_vals(master_data[:, i], master_label)
    n, hist_0, hist_1, hist_2 =  generate_labelhist(label_0, label_1, label_2, master_data[:, i])
    plt_hist(n, hist_0, hist_1, hist_2, f"{FEATURE_VARS[i]}")
# %% Basic model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 7)),
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
# %%
# fit model
NUM_EPOCHS = 20
BATCH_SIZE = 10
lim = 20 #len(train_data)
first = train_data[:lim]
model.fit(x = first, y = train_label[0:lim], validation_data = (test_data[0:lim], test_label[0:lim]), batch_size = BATCH_SIZE, epochs=NUM_EPOCHS)
# %%
plot_accuracy_loss(model, NUM_EPOCHS)
# %%
# Predict on test data
num_do = 10 #test_label.shape[0]
model_predictions = np.zeros(shape = (num_do, 3))
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
# Converting to DataFrame for easy access
#Only for user study, dataframe not needed for functions
lims = [0, len(model_predictions)]
first = pd.DataFrame(test_data[lims[0]:lims[1]], columns = FEATURE_VARS)
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
tf.keras.models.save_model( model, f"../models/{MODELTYPE}/{temp.tm_mon}-{temp.tm_mday}", overwrite=True,)
# %%
############## End of Regular File ################
# %%
#Load Previous Saved model
load_model_name = "12-3"
model = tf.keras.models.load_model(f"../models/{MODELTYPE}/{load_model_name}")
# %%
