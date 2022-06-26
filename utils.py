import copy
import numpy as np


def load_data(path):
    train, label = [], []
    f = open(path)
    for line in f.readlines():
        sample = []
        line = line.strip().strip('\t').split(',')
        label.append(line[-1])
        for i in range(len(line) - 1):
            sample.append(line[i])
        train.append(sample[1:])
    train = label_encoder(np.array(train), mode='train')
    label = label_encoder((np.array(label).reshape(-1, 1)), mode='label')
    label = [num[0] for num in label]

    return train, label


def combine(X, Y):
    data = copy.deepcopy(X)
    for i in range(len(X)):
        data[i] = list(X[i])
        data[i].append(Y[i])
    return data


def leaf(dataset):
    data = np.array(dataset)
    return np.mean(data[:, -1])


def split_tree(data, feature, split_val):
    set_L, set_R = [], []
    tmp_LX, tmp_LY, tmp_RX, tmp_RY = [], [], [], []
    for line in data:
        if line[feature] < split_val:
            tmp_LX.append(line[:-1])
            tmp_LY.append(line[-1])
        else:
            tmp_RX.append(line[:-1])
            tmp_RY.append(line[-1])
    set_L.append(tmp_LX)
    set_L.append(tmp_LY)
    set_R.append(tmp_RX)
    set_R.append(tmp_RY)
    return set_L, set_R


def is_number(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def label_encoder(data, mode=None):
    if mode == 'train':
        data = data[1:, 1:]
    else:
        data = data[1:]
    m, n = len(data), len(data[0])
    for col in range(n):
        if not is_number(data[1][col]):
            data_dict = {}
            seq = 0
            for i in range(m):
                if data[i][col] not in data_dict.keys():
                    data_dict[data[i][col]] = seq
                    seq += 1
                data[i][col] = data_dict[data[i][col]]
    data = data.tolist()
    for i in range(m):
        for j in range(n):
            if len(data[i][j]) > 0:
                data[i][j] = float(data[i][j])
            else:
                data[i][j] = float(0)
    return data

