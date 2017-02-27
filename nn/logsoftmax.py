from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module

import numpy as np


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()


    def forward(self, inp):
        self.output = np.log( np.exp(inp) / np.sum(np.exp(inp), axis=0) )
        return self.output
