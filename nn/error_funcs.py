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
from nn.func import Function


class SquaredError(Function):
    name = 'se'

    def get_value(self, t, y):
        return ((t - y).T @ (t - y)).flatten()[0] / 2.0

    def get_derivative(self, t, y):
        return -(t - y)


# def se(t, y):
#     return ((t - y).T @ (t - y)).flatten()[0] / 2.0
# se.name = 'se'
#
#
# def d_se(t, y):
#     return -(t - y)
# d_se.name = 'se'
#
#
# def cross_entropy(t, y):
#     return (-1.0) * np.sum(t * np.log(y) + (1.0 - t) * np.log(1 - y))
# cross_entropy.name = 'cross_entropy'
#
#
# def d_cross_entropy(t, y):
#     return (1.0 - t) / (1.0 - y) - t / y
# d_cross_entropy.name = 'cross_entropy'
