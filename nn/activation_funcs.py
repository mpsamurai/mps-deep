# The MIT License (MIT)
# Copyright (c) 2016 Junya Kaneko <jyuneko@hotmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
from copy import deepcopy
from nn.func import Function


class Logistic(Function):
    name = 'logistic'

    def get_value(self, s):
        return 1 / (1 + np.exp(-s))

    def get_derivative(self, y):
        assert isinstance(y, np.ndarray) and y.shape[1] == 1
        jacobian = np.zeros(shape=(y.shape[0], y.shape[0]))
        jacobian[np.diag_indices(y.shape[0])] = (y * (1 - y)).flatten()
        return jacobian


class Tanh(Function):
    name = 'tanh'

    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def get_value(self, s):
        return self._alpha * np.tanh(self._beta * s)

    def get_derivative(self, y):
        assert isinstance(y, np.ndarray) and y.shape[1] == 1
        derivative = np.zeros(shape=(y.shape[0], y.shape[0]))
        derivative[np.diag_indices(y.shape[0])] = self._alpha * self._beta * (1 - np.power(np.tanh(self._beta * y.flatten()), 2))
        return derivative


class Softmax(Function):
    name = 'softmax'

    def get_value(self, s):
        _s = s - np.max(s)
        exp_s = np.exp(_s)
        value = exp_s / np.sum(exp_s)
        value[value < 10e-7] = 10e-7
        value[value > 10e+7] = 10e+7
        return value / np.sum(value)

    def get_derivative(self, y):
        assert isinstance(y, np.ndarray) and y.shape[1] == 1
        y_diag = np.zeros(shape=(y.shape[0], y.shape[0]))
        np.fill_diagonal(y_diag, y.flatten())
        return y_diag - y @ y.T


class Rectifier(Function):
    def get_value(self, s):
        val = deepcopy(s)
        val[val < 0.0] = 0.0
        return val

    def get_derivative(self, y):
        val = deepcopy(y)
        val[val > 0.0] = 1.0
        val[val <= 0.0] = 0.0
        y_diag = np.zeros(shape=(y.shape[0], y.shape[0]))
        np.fill_diagonal(y_diag, val.flatten())
        return y_diag
