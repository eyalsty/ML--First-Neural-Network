import numpy as np


def minmax_params(dset):
    """
    assuming - 'dset' is the whole data set - including the labels !
    :param dset: the data set which we like to get it's min max norm parameters.
    :return:  two arrays of mins and denoms, which are the min-max norm params.
    """
    features = dset.shape[1]

    old_mins = np.zeros(features)
    denoms = np.zeros(features)

    for i in range(features):
        old_mins[i] = np.amin(dset[:, i])
        denoms[i] = np.amax(dset[:, i]) - old_mins[i]

        if denoms[i] == 0:
            denoms[i] = 1 / dset.shape[0]

    return old_mins, denoms


def minmax(dset, new_min, new_max, old_mins=None, denoms=None):
    """
    :param dset: the data which we like to norm.
    :param new_min: the new minimum for the norm.
    :param new_max: the new maximum for the norm.
    :param old_mins: optional - for the case of normalize the test set.
    :param denoms: optional - for the case of normalize the test set.
    :return: a copy of 'dset', only normalized.
    """
    _dset = np.copy(dset)

    # for the case where 'dset' is the train set, which includes the labels too.
    features = _dset.shape[1]

    if old_mins is None and denoms is None:

        for i in range(features):
            old_min = np.amin(_dset[:, i])
            denom = np.amax(_dset[:, i]) - old_min

            if denom == 0:
                denom = 1 / _dset.shape[0]

            value = (_dset[:, i] - old_min) / denom

            _dset[:, i] = value * (new_max - new_min) + new_min

        return _dset

    # the case that 'dset' is a test set, with no labels column.
    # features += 1
    for i in range(features):
        value = (_dset[:, i] - old_mins[i]) / denoms[i]

        _dset[:, i] = value * (new_max - new_min) + new_min

    return _dset


def zscore_params(dset):
    """
    :param dset: the data set which working on.
    :return: the params of the normalization on 'dset'.
    """

    features = dset.shape[1]

    old_means = np.zeros(features)
    old_stds = np.zeros(features)

    for i in range(features):
        old_means[i] = np.mean(dset[:, i])
        old_stds[i] = np.std(dset[:, i])

        if old_stds[i] == 0:
            old_stds[i] = 1 / dset.shape[0]

    return old_means, old_stds


def zscore(dset, train_means=None, train_stds=None):
    """
    :param dset: the data set which we like to normalize.
    :param train_means: (optional) the means of the features of the train set.
    :param train_stds: (optional) the stand' dev of the features of the train set.
    :return: a copy of 'dest', only z-scored normalized.
    """

    _dset = np.copy(dset)
    features = dset.shape[1]

    if train_means is None and train_stds is None:

        for i in range(features):
            mean = np.mean(_dset[:, i])
            stand_dev = np.std(_dset[:, i])

            if stand_dev == 0:
                stand_dev = 1 / dset.shape[0]

            _dset[:, i] = (_dset[:, i] - mean) / stand_dev

        return _dset

    for i in range(features):
        if train_stds[i] != 0:
            _dset[:, i] = (_dset[:, i] - train_means[i]) / train_stds[i]

    return _dset
