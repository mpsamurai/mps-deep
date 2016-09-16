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
from nn.activation_funcs import Logistic, Tanh, Softmax, Rectifier


class BaseLayer:
    name = 'base'

    def __init__(self, n_output, n_prev_output, f):
        """
        :param n_output: Number of this layer's output
        :param n_prev_output: Number of previous layer's output
        :param f: Activation function (callable)
        :param df: Derivative of the activation function (callable)
        """
        self._W = self._init_W(n_output, n_prev_output)
        self._b = self._init_b(n_output)
        self._f = f
        self._y = None
        self._delta = None

    def __str__(self):
        return self.name

    def _init_W(self, n_output, n_prev_output, **kwargs):
        """
        Way of initializing weight matrix W

        :param n_output: Number of this layer's output
        :param n_prev_output: Number of previous layer's output
        :param kwargs:
        :return: numpy ndarray object having dimension n_output * n_prev_output
        """
        return np.random.uniform(-1, 1, size=(n_output, n_prev_output))

    def _init_b(self, n_output, **kwargs):
        """
        Way of initializing bias

        :param n_output: Number of this layer's output
        :param kwargs:
        :return: numpy ndarray object having dimension n_output * 1
        """
        return np.random.uniform(-1, 1, size=(n_output, 1))

    @property
    def n_output(self):
        return self._W.shape[0]

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        self._W = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def ave_abs_W(self):
        return np.average(np.abs(self._W))

    @property
    def ave_W(self):
        return np.average(self._W)

    @property
    def y(self):
        return self._y

    @property
    def delta(self):
        return self._delta

    def propagate_forward(self, x):
        self._y = self._f.get_value(self._W @ x + self._b)
        return self._y

    def propagate_backward(self, next_delta, next_W):
        if next_W is not None:
            self._delta = self._f.get_derivative(self._y) @ next_W.T @ next_delta
        else:
            self._delta = self._f.get_derivative(self._y) @ next_delta
        return self._delta

    def update(self, prev_y, epsilon):
        Delta_W = self._delta @ prev_y.T
        self._W -= epsilon * Delta_W
        self._b -= epsilon * self._delta

    def to_json(self):
        return {'type': self.name, 'W': self._W.tolist(), 'b': self._b.tolist()}


class LogisticLayer(BaseLayer):
    name = 'logistic'

    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, Logistic())


class TanhLayer(BaseLayer):
    name = 'tanh'

    def __init__(self, n_output, n_prev_output, alpha, beta):
        super().__init__(n_output, n_prev_output, Tanh(alpha, beta))

    def to_json(self):
        data = super(TanhLayer, self).to_json()
        data['alpha'] = self._f.alpha
        data['beta'] = self._f.beta
        return data


class SoftmaxLayer(BaseLayer):
    name = 'softmax'

    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, Softmax())


class RectifierLayer(BaseLayer):
    name = 'rectifier'

    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, Rectifier())
