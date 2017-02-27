from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module

import numpy as np


class Linear(Module):
    def __init__(self, n_inp, n_outp):
        super(Linear, self).__init__()

        self.n_inp = n_inp
        self.n_outp = n_outp
        self.w = np.random.randn(n_inp, n_outp) * 0.01
        self.b = np.ones([n_outp])
        self.last_inp = None
        self.grad = None

    def forward(self, inp):
        self.last_inp = inp
        self.output = np.matmul(inp, self.w) + self.b
        return self.output

    def update_grad_input(self, grad_prev):
        self.grad_input = grad_prev.dot(self.w.T)
        #self.grad_input = self.w.dot(grad_prev.T)
        self.grad_input_w = np.matmul(grad_prev.T, self.last_inp)
        self.grad_input_b = grad_prev
        #print(self.grad_input_w)

    def update_parameters(self, gr=None):
        alpha = 0.1
        self.w -= alpha * self.grad_input_w.T
        self.b -= alpha * self.grad_input_b.mean(axis=0)
