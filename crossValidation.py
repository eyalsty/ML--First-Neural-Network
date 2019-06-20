import NeuralNet
from NeuralNet import NeuralNetwork
import numpy as np
import Norm as norm
import timeit



def cross_validate(network_shape, epochs_num, learn_rate, _groups_x, _groups_y):
    k = _groups_x.shape[0]
    _sum = 0
    results = np.zeros(k)
    for i in range(k):

        train_x = None
        train_y = None
        valid_x = np.copy(_groups_x[i])  # the validation set for th i'th iteration.
        valid_y = np.copy(_groups_y[i])

        net = NeuralNetwork(network_shape, epochs_num, learn_rate)

        for j in range(k):
            if j != i:
                # arrange the train set for the i'th iteration.
                if train_x is None:
                    train_x = np.copy(_groups_x[j])
                    train_y = np.copy(_groups_y[j])
                else:
                    train_x = np.concatenate((train_x, _groups_x[j]), axis=0)
                    train_y = np.concatenate((train_y, _groups_y[j]), axis=0)

        old_mins, denoms = norm.minmax_params(train_x)
        train_x = norm.minmax(train_x, 0, 1)
        valid_x = norm.minmax(valid_x, 0, 1, old_mins, denoms)

        net.train(train_x, train_y)
        results[i] = net.accuracy(valid_x, valid_y)

        old_mins, denoms = norm.minmax_params(train_x)
        train_x = norm.minmax(train_x, 0, 1)
        valid_x = norm.minmax(valid_x, 0, 1, old_mins, denoms)

    print(results)
    return np.average(results)


def plot_epochs(max_epochs, l_rate, net_shape):
    for epochs_num in range(max_epochs):
        start = timeit.default_timer()
        print("results of {} epochs:".format(epochs_num))
        average = cross_validate(net_shape, epochs_num, l_rate, _groups_x, _groups_y)
        print("Average Of:{}".format(average))
        stop = timeit.default_timer()
        print('Time Took: {} seconds '.format(stop - start))


samples = NeuralNet.load_samples("train_x")
labels = NeuralNet.load_labels("train_y")

n_groups = 5
max_epochs = 6
l_rate = 0.1
net_shape = [784, 24, 10]

_groups_x = np.copy(np.array_split(samples, n_groups))
_groups_y = np.copy(np.array_split(labels, n_groups))

plot_epochs(max_epochs, l_rate, net_shape)

net_shape = [784, 100, 10]
max_epochs = 15
l_rate = 0.111
plot_epochs(max_epochs, l_rate, net_shape)


