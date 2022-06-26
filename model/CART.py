from utils import *
from train_eval import err_cnt


class Node:
    """
    CART树节点定义
    """
    def __init__(self, feature=0, split_val=None, results=None, left=None, right=None):
        self.feature = feature
        self.split_val = split_val
        self.results = results
        self.left = left
        self.right = right


class CART_regression(object):
    """
    CART 回归树算法
    """
    def __init__(self, X, Y, min_sample, min_err):
        self.X = X
        self.Y = Y
        self.min_sample = min_sample
        self.min_err = min_err

    def forward(self):
        data = combine(self.X, self.Y)
        data = np.array(data)

        bestErr = err_cnt(data)
        bestCriteria = None
        bestSets = 0
        err_now = 0

        if len(data) <= self.min_sample or bestErr < self.min_err:
            return Node(results=leaf(data))

        for feat in range(len(data[0]) - 1):
            val_feat = np.unique(data[:, feat])
            for val in val_feat:
                set_L, set_R = split_tree(data, feat, val)
                comb_L = combine(set_L[0], set_L[1])
                comb_R = combine(set_R[0], set_R[1])
                err_now = err_cnt(comb_L) + err_cnt(comb_R)
                if len(comb_L) < 2 or len(comb_R) < 2:
                    continue
                if err_now < bestErr:
                    bestErr = err_now
                    bestCriteria = (feat, val)
                    bestSets = (set_L, set_R)

        if err_now > self.min_err:
            left = CART_regression(bestSets[0][0], bestSets[0][1], self.min_sample, self.min_err).forward()
            right = CART_regression(bestSets[1][0], bestSets[1][1], self.min_sample, self.min_err).forward()
            return Node(feature=bestCriteria[0], split_val=bestCriteria[1], left=left, right=right)
        else:
            return Node(results=leaf(data))



