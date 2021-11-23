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
BGROUND_Label = {0:[1]}
M_SIG_Label = {0:[0]}
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
master_data = log10(parent_data)
global_max = np.nanmax(master_data)
global_min = np.nanmin(master_data)
altered_min = global_min - 1 #normalized 0 reserved for NaN

def try_1(sep):
    """
    Current feature extraction version
    """
    inter = np.multiply(sep, -1)
    inter = shift(inter,1 - np.nanmin(inter))
    inter = np.log2(inter)
    inter = 1 - custom_norm(inter, -1.5, np.nanmax(inter)+ 1.5, 1.)
    return inter

def try_2(master_data):
    master_data = shift(master_data, 2 - global_min)
    master_data = nanlog(master_data, np.log(2)) 
    new_global_max = np.nanmax(master_data)
    master_data = custom_norm(master_data, 0, new_global_max)

master_data = try_1(master_data)
# %% 
# Print images
plot_data(master_data, master_label, shape = (5, 4), start = 0)

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

train_data, train_label, test_data, test_label = split_data(master_data, master_label, train_ratio=0.8)
# %%
# fit model
NUM_EPOCHS = 20
BATCH_SIZE = 10
model.fit(x = train_data, y = train_label, validation_data = (test_data, test_label), batch_size = BATCH_SIZE, epochs=NUM_EPOCHS)
# %%
plt.figure()
x_vals = np.arange(1, NUM_EPOCHS + 1)
plt.plot( x_vals, model.history.history['accuracy'], label = 'accuracy')
plt.plot( x_vals, model.history.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel('Accuracy')
plt.title("Accuracy over epochs")
plt.show()
plt.plot( x_vals, model.history.history['loss'], label = "loss")
plt.plot( x_vals, model.history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.title("Loss over epochs")
plt.show()
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
"""
test_acc = 0
for i in range(0, len(predicted_labels)):
    if predicted_labels[i] == test_label[i]:
        test_acc += 1
test_acc /= len(predicted_labels)
"""


# %%
label_0, label_1, label_2 = [], [], []
for i in range(0, len(disc_vals)):
    if test_label[i] == 0:
        label_0.append(disc_vals[i])
    elif test_label[i] == 1:
        label_1.append(disc_vals[i])
    elif test_label[i] == 2:
        label_2.append(disc_vals[i])

label_0 = np.array(label_0)
label_1 = np.array(label_1)
label_2 = np.array(label_2)

def generate_outline_hist(bins, height):
    x_res = np.zeros(shape = (2*len(height) + 2))
    y_res = np.zeros(shape = (2*len(height) + 2))
    width = bins[1] - bins[0]
    #Start case
    x_res[0] = bins[0]
    y_res[0] = 0
    index_num = 1
    for i in range(0, len(n) - 1):
        x_res[index_num] = bins[i]
        x_res[index_num + 1] = bins[i + 1]
        y_res[index_num] = height[i]
        y_res[index_num + 1] = height[i]
        index_num += 2
    x_res[-1] = bins[-1]
    y_res[-1] = 0
    return x_res, y_res

num_bins = 40
disc_range = (min(disc_vals), max(disc_vals))
hist_0, n = np.histogram(label_0, bins = num_bins, range = disc_range)
hist_1, n = np.histogram(label_1, bins = num_bins, range = disc_range)
hist_2, n = np.histogram(label_2, bins = num_bins, range = disc_range)

#%%
def plt_log10(x):
    if x <= 0:
        return 0
    return np.log10(x)
plt_log10 = np.vectorize

plt.figure(figsize = (6, 6))
x, y_0 = generate_outline_hist(n, hist_0)
x, y_1 = generate_outline_hist(n, hist_1)
x, y_2 = generate_outline_hist(n, hist_2)
plt.semilogy(x, y_0/y_0.sum(), color = "blue", linestyle = "solid", label = "Label 0")
plt.semilogy(x, y_1/y_1.sum(), color = "green", linestyle = "dashed", label = "Label 1")
plt.semilogy(x, y_2/y_2.sum(), color = "red", linestyle = "dashdot", label = "Label 2")
#plt.xticks(n)
plt.xlabel("D")
plt.ylabel("Label fraction")
plt.tight_layout()
plt.legend()

# %%
def eff_rej_calc(label_0, label_1, label_2, rej = True):
    if rej:
        rej_1 = [(label_1 < i).sum()/len(label_1) for i in label_0]
        rej_2 = [(label_2 < i).sum()/len(label_2) for i in label_0]
        x = [(label_0 > i).sum()/len(label_0) for i in label_0]
        return x, rej_1, rej_2
    else:
        rej_1 = [(label_1 >= i).sum()/len(label_1) for i in label_0]
        rej_2 = [(label_2 >= i).sum()/len(label_2) for i in label_0]
        x = [(label_0 > i).sum()/len(label_0) for i in label_0]
        return x, rej_1, rej_2  

# %%
#Obtain Rejection rates
label_0.sort()
rej = True

x_frac, y_1, y_2 = eff_rej_calc(label_0, label_1, label_2, rej = rej)
plt.figure(figsize = (7, 7))
plt.plot(x_frac, y_1, color = "green", linestyle = "dashed", label = "Label 1")
plt.plot(x_frac, y_2, color = "red", linestyle = "dashdot", label = "Label 2" )
if rej:
    plt.ylabel("jet rejection")
    plt.title("Jet rejection versus hbb efficiency")
else:
    plt.ylabel("jet efficiency")
    plt.title("Jet efficiency versus hbb efficiency")
plt.xlabel("Hbb jet efficiency")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
#%%
#Calculate predicted label based on discriminant
predicted_labels = []

# %%
def calc_prec_recall(true_label, prediction, choose_label):
    tp = ((prediction == choose_label) & (true_label == choose_label)).sum()
    fp = ((prediction == choose_label) & (true_label != choose_label)).sum()
    fn = ((prediction != choose_label) & (true_label == choose_label)).sum()
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
print(confusion_mat)
# %%
temp = time.localtime()
tf.keras.models.save_model( model, f"../models/cnn/{temp.tm_mon}-{temp.tm_mday}", overwrite=True,)

# %%
load_model_name = "11-13"
model = tf.keras.models.load_model(f"../models/cnn/{load_model_name}")
# %%
np.save("model_preds_11-23.npy", model_predictions)
# %%
model_predictions = np.load("model_preds_11-23.npy")
# %%
