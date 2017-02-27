from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module

import numpy as np


class MSE(Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x, y):
        self.last_inp = x
        output = ((x - y) ** 2).mean()#axis=1)
        return output

    def update_grad_input(self, t):
        self.grad_input = -2.0 / (t.shape[0] * t.shape[1]) * (t - self.last_inp)#.mean(axis=1)
        pass
