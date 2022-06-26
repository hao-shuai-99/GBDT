import argparse
from model.GBDT import GBDT_RT
from model.CART import CART_regression
import random
from utils import *
from train_eval import train_tree, test

parser = argparse.ArgumentParser("CART and GBDT model.")

# CART param
parser.add_argument('--min_sample', default=20, type=int, help="Min sample for leaf nodes")
parser.add_argument('--min_err', default=20, type=float, help="Minimum reduction of node splitting loss")

# GBDT param
parser.add_argument('--n_estimates', default=10, type=int, help="The number of CART tree in GBDT.")
parser.add_argument('--learning_rate', default=0.1, type=float, help="learning rate")

# model param
parser.add_argument('--model', default='GBDT', type=str, help='GBDT or CART')
parser.add_argument('--path', default='./data/regression_train.csv', help="dataset path")
parser.add_argument('--test_size', default=0.1, type=float, help="test size of dataset")
args = parser.parse_args()


def main(args):
    train, label = load_data(args.path)
    test_size = int(len(train) * args.test_size)
    train, label = np.array(train), np.array(label)
    train_x, train_y = train[test_size:, :], label[test_size:]
    test_x, test_y = train[:test_size, :], label[:test_size]
    model = None
    if args.model == 'GBDT':
        model = GBDT_RT(train_x.tolist(), train_y.tolist(), args)
    if args.model == 'CART':
        model = CART_regression(train_x.tolist(), train_y.tolist(), args.min_sample, args.min_err)

    trained_model = train_tree(model, args)
    test(trained_model, args, test_x.tolist(), test_y.tolist())
    pass


if __name__ == '__main__':
    random.seed(2022)
    np.random.seed(2022)

    main(args)
