import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from file_support import *
def plot_data(data, labels, shape = (5, 4), start = 0): 
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
        
def shuffle_arrays(inputs, size = SIDE_SIZE,  seed = 0):
    """
    Shuffles a list of arrays in a single iteration (O(n) time).

    Generates a random variable that chooses next array to be popped 
    from, corresponding to a weighted set of thresholds.
    """
    np.random.seed(seed)
    start_elems = [0 for i in inputs]
    rem_sizes = [0 for i in inputs]

    def update_rem():
        nonlocal rem_sizes
        rem_sizes = [inputs[i].shape[0] - start_elems[i] for i in range(0, len(inputs))]

    thresholds = [1 for i in inputs]
    def calc_threshold():
        update_rem()
        total = sum(rem_sizes)
        floor = 0
        for i in range(0, len(thresholds)):
            temp = np.divide(rem_sizes[i], total) if rem_sizes[i] != 0 else 0
            floor += temp
            thresholds[i] = floor

    calc_threshold()
    main_output = np.zeros((sum(rem_sizes), size, size))
    main_elem = 0
    while sum(rem_sizes) != 0:
        rand_var = np.random.random()
        choose_array = None
        for i in range(0, len(thresholds)):
            if thresholds[i] > rand_var and rem_sizes[i]:
                choose_array = i
                break
        main_output[main_elem] = inputs[choose_array][start_elems[choose_array]]
        start_elems[i] += 1
        main_elem += 1
        calc_threshold()
    if main_elem != main_output.shape[0]:
        print(f"Disparity in index: interated main index: {main_elem}, original calculated size: {main_output.shape[0]}")
        return 1, main_output
    return 0, main_output
