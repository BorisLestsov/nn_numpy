from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import nn


def main():
    np.random.seed(1)

    model = nn.Sequential()
    model.add(nn.Linear(2, 5))
    model.add(nn.Sigmoid())
    model.add(nn.Linear(5, 1))
    #model.add(nn.Sigmoid())
    model.set_metric(nn.MSE())

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model.fit(x, y, 5000, 1)


if __name__ == '__main__':
    main()