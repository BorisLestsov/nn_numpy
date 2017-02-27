from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module

import numpy as np


class NegLogLiklehood(Module):
    def __init__(self):
        super(NegLogLiklehood, self).__init__()

    def forward(self, x, y):
        self.output = -np.sum(np.dot(y, np.log(x)) + np.dot(1 - y, np.log(1 - x)))
        return self.output

