import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
def plot_data(data, labels, shape = (5, 4), start = 0): 
    #shape - (row, colulm)
    plt.figure(figsize = (3 * shape[1], 3 * shape[2]))
    for elem in range(0, shape[0] * shape[1]):
        plt.subplot(shape[0], shape[1], elem)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(data[elem + start])
        plt.xlabel(f"{labels[elem + start]}")

def plot_performance(data, labels, predictions, shape = (5, 4), start = 0):
    plt.figure(figsize = (3 * shape[1], 3 * shape[2]))
    for elem in range(0, shape[0] * shape[1]):
        plt.subplot(shape[0], shape[1], elem)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(data[elem + start])
        str_val = f"{predictions[elem + start]}({labels[elem + start]})"
        plt.xlabel(str_val, color = 'green' if np.argmax(predictions[elem + start]) == labels[elem + start] else 'red')

"""
show images regular data
image visualization of performance
confusion matrix
"""


