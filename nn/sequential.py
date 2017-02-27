from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module

import numpy as np

class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.layers = []
        self.metric = None

    def add(self, module):
        self.layers.append(module)

    def set_metric(self, module):
        self.metric = module

    def remove(self, module):
        raise NotImplementedError('implement remove to sequential!')

    def fit(self, x, y, epoches, b_size):
        for epoch in range(epoches):
            loss = 0.0
            n_batches = int(x.shape[0] / b_size)
            for batch_i in range(n_batches):
                i_start, i_end = batch_i*b_size, (batch_i + 1)*b_size
                res = self.forward(x[i_start : i_end])
                loss += self.metric.forward(res, y[i_start : i_end])
                self.backward(y[i_start : i_end])

            print(loss)


    def predict(self, x):
        return self.forward(x)

    def forward(self, inputs):
        prev_outp = inputs
        for L in self.layers:
            prev_outp = L.forward(prev_outp)
        return prev_outp

    def backward(self, y):
        if self.metric is None:
            raise Exception("Metric is not set")

        grad = self.metric.backward(y)
        for L in reversed(self.layers):
            grad = L.backward(grad)




