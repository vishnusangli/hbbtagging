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
#from sklearn.model_selection import train_test_split
# %%
#Directory and label file selection
BGROUND_Label = {0:[1]}
M_SIG_Label = {0:[0]}
I_SIG_Label = {0:[2]}
MODELTYPE = 'cnn'
# Retrieve Datasets
misc_vals, parent_data, master_label, jet_codes = get_files(BGROUND_Label, M_SIG_Label, I_SIG_Label, misc_features=8, seed = 0)
count_labels(master_label)
# %%
# Feature engineering
master_data = jet_img_FeatureEngineering.current(parent_data)
# %% 
# Print images
plot_data(master_data, master_label, shape = (5, 4), start = 0)

# %%
#Create custom metric, loss functions?

# %% Basic model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 30)),
    tf.keras.layers.Dense(128, activation=tf.nn.softplus),
    tf.keras.layers.Dense(60, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', "sparse_categorical_accuracy"])
# %% 
# Train/test split data
train_data, train_label, train_code, test_data, test_label, test_code = ter_split_data(master_data, master_label, jet_codes, train_ratio=0.8)
# %%
# fit model
NUM_EPOCHS = 20
BATCH_SIZE = 10
model.fit(x = train_data, y = train_label, validation_data = (test_data, test_label), batch_size = BATCH_SIZE, epochs=NUM_EPOCHS)
# %%
plot_accuracy_loss(model, NUM_EPOCHS)
# %%
#Test Accuracy
test_loss, test_accuracy = model.evaluate(train_data, train_label)
print(f"Accuracy on test dataset: {test_accuracy}")
# %%
# Predict on test data
# TODO add f1 metric
num_do = 100 #test_label.shape[0]
model_predictions = np.zeros(shape = (test_label.shape[0], 3))
for i in range(0, test_label.shape[0]):
    model_predictions[i] = model.predict(np.reshape(test_data[i], (1, 30, 30)) ) 
    if i % 50 == 0:
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

naive_pred = lambda x: np.array([np.argamx(i) for i in x])
disc_vals = np.array([disc_var(model_predictions[i], 0.9) for i in range(0, model_predictions.shape[0])])

#Performance metrics
# accuracy, 
# precision, recall, F1 on label 0?

#test_accuracy = (predicted_labels == test_label).sum()/predicted_labels.shape[0]
# %%
label_0, label_1, label_2 = sort_disc_vals(disc_vals, test_label)
n, hist_0, hist_1, hist_2 =  generate_labelhist(label_0, label_1, label_2, disc_vals)
plt_hist(n, hist_0, hist_1, hist_2, f"Discriminating variable for {MODELTYPE}")
# %%
#Obtain Rejection rates
plot_rej_rates(label_0, label_1, label_2, MODELTYPE) 
# %%
#Calculate predicted label based on discriminant
predicted_labels = []


# %%
#Confusion Matrix
confusion_mat = tf.math.confusion_matrix(
    test_label, predicted_labels, num_classes=None, weights=None, dtype=tf.dtypes.int32,
    name=None
)
plt.imshow(confusion_mat, )
plt.xticks([0, 1, 2])
plt.xlabel("Prediction")
plt.yticks([2, 1, 0])
plt.ylabel("Label")
plt.colorbar()
plt.title("Confusion Matrix")
print(confusion_mat)
# %%
temp = time.localtime()
tf.keras.models.save_model( model, f"/global/homes/v/vsangli/starters/models/cnn/{temp.tm_mon}-{temp.tm_mday}", overwrite=True,)

# %%
############## End of Regular File ################
"""
# %%
load_model_name = "11-13"
model = tf.keras.models.load_model(f"/global/homes/v/vsangli/starters/models/cnn/{load_model_name}")
# %%
np.save("model_preds_11-23.npy", model_predictions)
# %%
model_predictions = np.load("model_preds_11-23.npy")
# %%
"""