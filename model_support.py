import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from file_support import *

def shift(val, lim):
    if np.isnan(val):
        return np.NaN
    return val + lim
shift = np.vectorize(shift)

def custom_norm(val, min, gap, nanval = 0.):
    if np.isnan(val):
        return nanval
    #val = val - min
    else:
        return np.divide(np.subtract(val, min) , gap)
custom_norm = np.vectorize(custom_norm)

def nanlog(val, e):
    """
    e is log(given val)
    """
    if val <= 0 or np.isnan(val):
        return np.NaN
    return np.log(val)/e
nanlog = np.vectorize(nanlog)

def filter_ignore(data, val = 0, ignore_nan = False):
    values = []
    for i in data.flatten():
        if ignore_nan:
            if not np.isnan(i):
                values.append(i)
        elif val != i:
            values.append(i)
    return values

def plot_data(data, labels, shape = (5, 4), start = 0): 
    fig, ax = plt.subplots(shape[0], shape[1], figsize = (15, 15), sharey = True)
    fig.patch.set_facecolor('grey')
    for elem in range(1, (shape[0] * shape[1]) + 1):
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
        
def shuffle_arrays(inputs, secondaries, shape = (SIDE_SIZE, SIDE_SIZE),  seed = 0):
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
    temp = sum(rem_sizes)
    main_output = np.zeros((temp, *shape))
    main_secondary = np.empty(shape = (temp, secondaries[0][0].shape[0]), dtype=type(secondaries[0][0]))
    main_elem = 0
    while sum(rem_sizes) != 0:
        rand_var = np.random.random()
        choose_array = None
        for i in range(0, len(thresholds)):
            if thresholds[i] > rand_var and rem_sizes[i]:
                choose_array = i
                break
        main_output[main_elem] = inputs[choose_array][start_elems[choose_array]]
        main_secondary[main_elem] = secondaries[choose_array][start_elems[choose_array]]
        start_elems[i] += 1
        main_elem += 1
        calc_threshold()
    if main_elem != main_output.shape[0]:
        print(f"Disparity in index: interated main index: {main_elem}, original calculated size: {main_output.shape[0]}")
        return 1, main_output
    return 0, main_output, main_secondary

def split_data(data, labels, train_ratio = 0.8):
    """
    lower bound value
    """
    train_elems = int(np.multiply(data.shape[0], train_ratio))
    train_data, train_label = data[0:train_elems], labels[0:train_elems]
    test_data, test_labels = data[train_elems:], labels[train_elems:]
    print(f"{train_data.shape[0]}:{test_data.shape[0]}")
    return train_data, train_label, test_data, test_labels

def count_labels(labels):
    to_return = {0:0, 1:0, 2:0}
    for i in labels:
        to_return[int(i)] += 1
    return to_return

def generate_outline_hist(bins, height):
    x_res = np.zeros(shape = (2*len(height) + 2))
    y_res = np.zeros(shape = (2*len(height) + 2))
    width = bins[1] - bins[0]
    #Start case
    x_res[0] = bins[0]
    y_res[0] = 0
    index_num = 1
    for i in range(0, len(bins) - 1):
        x_res[index_num] = bins[i]
        x_res[index_num + 1] = bins[i + 1]
        y_res[index_num] = height[i]
        y_res[index_num + 1] = height[i]
        index_num += 2
    x_res[-1] = bins[-1]
    y_res[-1] = 0
    return x_res, y_res

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

def calc_prec_recall(true_label, prediction, choose_label):
    tp = ((prediction == choose_label) & (true_label == choose_label)).sum()
    fp = ((prediction == choose_label) & (true_label != choose_label)).sum()
    fn = ((prediction != choose_label) & (true_label == choose_label)).sum()
    return np.divide(tp, tp + fp), np.divide(tp, tp + fn)

def plot_accuracy_loss(model, num_epochs):
    plt.figure()
    x_vals = np.arange(1, num_epochs + 1)
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

def generate_labelhist(disc_vals, test_label):
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

    num_bins = 40
    disc_range = (min(disc_vals), max(disc_vals))
    hist_0, n = np.histogram(label_0, bins = num_bins, range = disc_range)
    hist_1, n = np.histogram(label_1, bins = num_bins, range = disc_range)
    hist_2, n = np.histogram(label_2, bins = num_bins, range = disc_range)
    return n, hist_0, hist_1, hist_2

def plt_hist(n, hist_0, hist_1, hist_2):
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
    plt.show()