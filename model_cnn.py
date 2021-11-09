# %% Imports 
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from file_support import *
from model_support import *
import math
#from sklearn.model_selection import train_test_split
# %%
#Directory and label file selection, Creation of Data Object
BGROUND_Label = {0:[1, 2, 3]}
M_SIG_Label = {0:[0, 1, 2, 3]}
I_SIG_Label = {0:[1, 2, 3]}
data_master = EventData()
# %% 
# Retrieve Datasets
file_obj = EventData(BGROUND_Label, M_SIG_Label, I_SIG_Label)
master_data, master_label = [], []
# %% 
# Feature engineering
#
#  Log10 to be applied
# normalization
# NaN imputation

# %% Print images
plot_data(master_data, master_label, shape = (5, 4), start = 0)

# %% Basic model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, SIDE_SIZE, SIDE_SIZE)),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.softmax),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])
#4 output neurons with softmax or adjusted perceptron threshold (messed idea but worth if viable computation availability)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# %% 
# Train/test split data
BATCH_SIZE = 30
train_data, train_label, test_data, test_label = 0, 0, 0, 0

# %%
# fit model
model.fit((train_data, train_label), epochs=20) #steps_per_epoch

# %%
test_loss, test_accuracy = model.evaluate((train_data, train_label))
print(f"Accuracy on test dataset: {test_accuracy}")
# %%
# Predict on test data
model_predictions = []
# %%
#Performance metrics
# accuracy, 
# precision, recall, F1 on label 0?
# %%
#Confusion Matrix
confusion_mat = tf.math.confusion_matrix(
    test_label, model_predictions, num_classes=None, weights=None, dtype=tf.dtypes.int32,
    name=None
)
plt.imshow(confusion_mat, )
plt.xticks(np.arange(0, 4, 1))
plt.xlabel("Prediction")
plt.yticks(np.arange(0, 4, 1))
plt.ylabel("Label")
plt.title("Confusion Matrix")