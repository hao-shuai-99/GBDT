import numpy as np
import time


def train_tree(model, args):
    start = time.time()
    model = model.forward()
    end = time.time()
    print("train spend {0} minutes, {1}seconds.".format(int((end - start) // 60), int((end - start) % 60)))
    return model


def test(model, args, test_x, test_y):
    predict = []
    if args.model == 'CART':
        for ind, sample in enumerate(test_x):
            pred = CART_predict(sample, model)
            predict.append(pred)
    if args.model == 'GBDT':
        predict = GBDT_predict(model, test_x, test_y)
    error = err_cal(test_y, predict)
    print("test error = %f" % error)


def CART_predict(sample, tree):
    if tree.results is not None:
        return tree.results
    else:
        val_sample = sample[tree.feature]
        if val_sample < tree.split_val:
            branch = tree.left
        else:
            branch = tree.right
    return CART_predict(sample, branch)


def GBDT_predict(trees, test_x, test_y):
    predicts = []
    for i in range(len(test_x)):
        pre_y = np.mean(test_y)
        for tree in trees:
            pre_y += CART_predict(test_x[i], tree)
        predicts.append(pre_y)
    return predicts


def err_cal(test_y, predict):
    error = 0
    for i in range(len(test_y)):
        error += pow(test_y[i] - predict[i], 2)
    return error / len(test_y)


def err_cnt(dataset):
    if len(dataset) == 0:
        return 0
    dataset = np.array(dataset)
    return np.var(dataset[:, -1]) * np.shape(dataset)[0]

