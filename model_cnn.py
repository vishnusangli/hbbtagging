# %% Imports 
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from file_support import *
from model_support import *
import math
import time
#from sklearn.model_selection import train_test_split
# %%
#Directory and label file selection
BGROUND_Label = {0:[1, 2]}
M_SIG_Label = {0:[0, 2]}
I_SIG_Label = {0:[2]}
MODELTYPE = 'cnn'
# Retrieve Datasets
dataset_arr = []
label_arr = []
for ev_type in [BGROUND_Label, M_SIG_Label, I_SIG_Label]:
    if ev_type == BGROUND_Label:
        ev_dir = BGROUND
    elif ev_type == M_SIG_Label:
        ev_dir = M_SIG
    else:
        ev_dir = I_SIG
    for event in ev_type.keys():
        for label in ev_type[event]:
            curr_arr = np.load(f"{DATA_DIR}/{ev_dir[event]}/label_{label}.npy")
            curr_label = np.full((curr_arr.shape[0], 1), label)

            dataset_arr.append(curr_arr)
            label_arr.append(curr_label)


success, parent_data, master_label = shuffle_arrays(dataset_arr, label_arr)
# %% 
# Feature engineering
#
#  Log10 to be applied
# normalization
# NaN imputation
master_data = log10(parent_data)
global_max = np.nanmax(master_data)
global_min = np.nanmin(master_data)
altered_min = global_min - 1 #normalized 0 reserved for NaN

# %%
sep = master_data[0]
def try_1(sep):
    inter = np.multiply(sep, -1)
    inter = shift(inter,1 - np.nanmin(inter))
    inter = np.log2(inter)
    inter = 1 - custom_norm(inter, -1.5, np.nanmax(inter)+ 1.5, 1.)
    return inter

master_data = try_1(master_data)
def try_2(master_data):
    master_data = shift(master_data, 2 - global_min)
    master_data = nanlog(master_data, np.log(2)) 
    new_global_max = np.nanmax(master_data)
    master_data = custom_norm(master_data, 0, new_global_max)
# %% Print images
plot_data(master_data, master_label, shape = (5, 4), start = 0)

# %% Basic model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 30)),
    tf.keras.layers.Dense(128, activation=tf.nn.softplus),
    tf.keras.layers.Dense(60, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
#4 output neurons with softmax or adjusted perceptron threshold (messed idea but worth if viable computation availability)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', "sparse_categorical_accuracy"])
# %% 
# Train/test split data
BATCH_SIZE = 30 #
train_data, train_label, test_data, test_label = split_data(master_data, master_label, train_ratio=0.8)
# %%
# fit model
model.fit(x = train_data, y = train_label, epochs=20) 

# %%
test_loss, test_accuracy = model.evaluate(train_data, train_label)
print(f"Accuracy on test dataset: {test_accuracy}")
# %%
# Predict on test data
model_predictions = []
for i in range(0, len(test_label)):
    model_predictions.append( model.predict( np.reshape(test_data[i], (1, 30, 30)) ) )
    if i % 500 == 0:
        print(f"{i} Done {model_predictions[-1]}")
# %%

predicted_labels = []
for i in model_predictions:
    predicted_labels.append(np.argmax(i))
predicted_labels = np.array(predicted_labels)
#Performance metrics
# accuracy, 
# precision, recall, F1 on label 0?
test_accuracy = (predicted_labels == test_label).sum()/predicted_labels.shape[0]
test_acc = 0
for i in range(0, len(predicted_labels)):
    if predicted_labels[i] == test_label[i]:
        test_acc += 1
test_acc /= len(predicted_labels)

# %%
def calc_prec_recall(label, prediction, choose_label):
    tp = 0
    fp = 0
    fn = 0
    for i in range(0, len(label)):
        if label[i] == choose_label:
            if prediction[i] == choose_label:
                tp += 1
            else:
                fn += 1
        elif prediction[i] == choose_label:
            fp += 1
    return np.divide(tp, tp + fp), np.divide(tp, tp + fn)
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
# %%
import time
temp = time.localtime()
tf.keras.models.save_model( model, f"../models/cnn/{temp.tm_mon}-{temp.tm_mday}", overwrite=True,)

# %%
to_return = {0:0, 1:0, 2:0}
for i in train_label:
    try:
        to_return[int(i)] += 1
    except IndexError:
        err.append(i)