from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module

import numpy as np

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()


    def forward(self, inp):
        self.output = 1.0 / (1 + np.exp(-inp))
        return self.output

    def update_grad_input(self, grad):
        self.grad_input = (self.output * (1 - self.output))*(grad)
        pass
