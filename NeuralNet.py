import numpy as np
import Norm as norm


def ReLU(z):
    return np.maximum(0, z)


def deriv_ReLU(z):
    copy = np.copy(z)

    copy[copy > 0] = 1
    copy[copy <= 0] = 0

    return copy


def softmax(z_last):
    # copy = np.copy(z_last)
    # stable = np.max(copy)
    # copy = copy - stable
    #
    # # return the output of the softmax function on each entry of the net output.
    # denom = np.sum(np.exp(copy))
    # return np.exp(copy) / denom
    z = z_last - np.max(z_last)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    s_max = numerator / denominator
    return s_max


def deriv_NLL(vs, label):
    """
    :param vs: an array of the activations for each layer.
    :param label: the label array for a specific example.
    :return: the derivative of the N.L.L function w.r.t the last
    weight matrix, and same w.r.t the bias vector.
    also notice, that the last activation function is 'softmax'.
    """
    y = np.copy(label)
    a1 = (vs[-1] - y)
    a2 = (vs[-2]).T

    # derivative w.r.t w & derivative w.r.t b, respectively.
    return np.dot(a1, a2), a1


def load_samples(path):
    return np.loadtxt(path)


def load_labels(path, dim=10):
    """
    :param path: the path for the label set file.
    :param dim: number of classes.
    :return: an ndarray of vectors, each vector is a label column (!)
    vector, s.t all of the entries are 0, except the right class
    entry .
    """
    data = np.loadtxt(path)
    labels = []
    for prediction in data:
        label = np.zeros((dim, 1))
        label[int(prediction)] = 1

        labels.append(label)

    return np.asarray(labels)


def split_dset(samples, labels, n_groups):
    copy1 = np.array(np.array_split(samples, n_groups))
    copy2 = np.array(np.array_split(labels, n_groups))

    return copy1, copy2


class NeuralNetwork:

    def __init__(self, net_shape, epochs=3, learning_rate=0.1):
        """
        saves the network shape and the number of layers. also initialize the biases
        and weights.
        :param net_shape: the network shape. example - [2, 2, 1] a 3 layer network,
        with 2 dim input layer, 2 dim input layer and 1 dim output layer.
        """
        self.net_shape = net_shape
        self.num_layers = len(net_shape)
        self.epochs = epochs
        self.l_rate = learning_rate

        # 2-dim array. each array contain the biases for the next layer.
        self.biases = np.array([(np.random.rand(y, 1)) for y in net_shape[1:]])

        ''' 
        array of matrices. each matrix represent the weight 
        matrix for crossing from one layer to the next one.
        for example - move from 3-dim layer to 4-dim layer, then
        the matrix shape will be = (4, 3) .
        '''
        self.weights = np.array([(np.random.rand(y, x))
                                 for x, y in zip(net_shape[:-1], net_shape[1:])])

    def forward_prop(self, x):
        """
        :param x: a single example from the training set.
        :return: two arrays - the first holds the activations products per layer,
        and the second holds the z's ( = w * a +b) per layer.
        doing so to get the parameters for the backprop algorithm.
        """
        v = np.copy(x)[:, np.newaxis]  # arrange as column vector.
        ws_num = self.num_layers - 1
        vs = [v]
        zs = [v]

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, v) + b
            z = z/255

            if i == ws_num - 1:
                v = softmax(z)  # the last layer
            else:
                v = ReLU(z)

            zs.append(z)
            vs.append(v)

        return np.asarray(vs), np.asarray(zs)

    def backprop(self, vs, zs, Y):

        dw_last, db_last = deriv_NLL(vs, Y)
        dws = [np.zeros(w.shape) for w in self.weights]
        dbs = [np.zeros(b.shape) for b in self.biases]

        dws[-1] = dw_last
        dbs[-1] = db_last

        delta = db_last
        for i in range(2, self.num_layers):
            deriv_f = deriv_ReLU(zs[-i])
            delta = deriv_f * np.dot((self.weights[-i + 1]).T, delta)

            dws[-i] = np.dot(delta, (vs[-i - 1]).T)
            dbs[-i] = delta
            # if i == 2:
            #     deriv = deriv_ReLU(zs[-i])
            #     delta = np.dot(self.weights[-1].T, delta)
            #     dws[-i] = np.dot(deriv * delta, vs[-i - 1].T)
            #     dbs[-i] = delta
            # else:
            #     deriv = deriv_ReLU(zs[-i])
            #
            #     deriv2 = deriv_ReLU(zs[-i + 1])
            #     delta = np.dot(self.weights[-i + 1].T, deriv2 * delta)
            #     dws[-i] = np.dot(deriv * delta, vs[-i - 1].T)
            #     dbs[-i] = deriv * delta

        return np.asarray(dws), np.asarray(dbs)

    def update(self, dws, dbs):

        for i in range(self.num_layers - 1):
            self.weights[i] = self.weights[i] - self.l_rate * dws[i]
            self.biases[i] = self.biases[i] - self.l_rate * dbs[i]

    def train(self, samples, labels):

        for i in range(self.epochs):
            for x, y in zip(samples, labels):

                vs, zs = self.forward_prop(x)
                dws, dbs = self.backprop(vs, zs, y)

                self.update(dws, dbs)

    def predict(self, x):

        v = np.copy(x)[:, np.newaxis]
        ws_num = self.num_layers - 1

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, v) + b

            if i == ws_num - 1:
                v = softmax(z)
            else:
                v = ReLU(z)

        return np.argmax(v)

    def accuracy(self, test_X, test_Y):
        error = 0
        size = len(test_X)

        for x, y in zip(test_X, test_Y):

            correct = np.argmax(y)
            prediction = self.predict(x)

            if correct != prediction:
                error += 1

        acc = (1 - float(error / size)) * 100
        return acc
        # print("accuracy = " + str(acc) + " %")


# net = NeuralNetwork([784, 24, 10])
# samples = load_samples("train_x")
# labels = load_labels("train_y")
# xs, ys = split_dset(samples, labels, 2)
#
# train_x, train_y = xs[0], ys[0]
# test_x, test_y = xs[1], ys[1]
#
# train_x = norm.minmax(train_x, 0, 1)
#
# net.train(train_x, train_y)
# net.accuracy(train_x, train_y)


# def load_samples_check(path):
#
#     """
#     for loading the samples, and returning it as an np Array.
#     """
#     with open(path, "r") as f:
#         arr = [[str(num) for num in line.split(',')] for line in f]
#     np_arr = np.copy(arr)
#     np_arr = np_arr[:, 1:].astype(float)  # gets array of all of the numeric values.
#     return np_arr
# samples = load_samples_check("ex2_set/train_x.txt")
# labels = load_labels("ex2_set/train_y.txt", 3)
#
# xs, ys = split_dset(samples, labels, 2)
# train_x, train_y = xs[0], ys[0]
# test_x, test_y = xs[1], ys[1]
#
# net = NeuralNetwork([7, 10, 3])
# net.train(train_x, train_y)
# net.accuracy(train_x, train_y)
