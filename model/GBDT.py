import matplotlib.pyplot as plt
from model.CART import CART_regression
from train_eval import CART_predict
import numpy as np


class GBDT_RT(object):
    def __init__(self, train, label, args):
        self.trees = None  # 存储生成的多棵回归树
        self.learn_rate = None  # 学习率
        self.init_val = None  # 初始值
        self.train = train
        self.label = label
        self.args = args

    def get_residuals(self, y, y_hat):
        y_residuals = []
        for i in range(len(y)):
            y_residuals.append(y[i] - y_hat[i])
        return y_residuals

    def forward(self):

        self.trees = []
        n = len(self.label)
        self.init_val = sum(self.label) / n
        y_hat = [self.init_val] * n
        y_residuals = self.get_residuals(self.label, y_hat)

        for k in range(self.args.n_estimates):
            tree = CART_regression(self.train, y_residuals, self.args.min_sample, self.args.min_err)
            root = tree.forward()
            for i in range(len(self.train)):
                res_hat = CART_predict(self.train[i], root)
                y_hat[i] += self.args.learning_rate * res_hat
            y_residuals = self.get_residuals(self.label, y_hat)
            self.trees.append(root)
        return self.trees




