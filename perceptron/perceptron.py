__author__ = 'Kai'

"""
实现一个感知器。
"""
import random
from decimal import Decimal


def round_float(value):
    return float(Decimal(value).quantize(Decimal('0.01')))


def random_float():
    return round_float(random.uniform(-0.5, 0.5))


def step_activation(value):
    if value < 0:
        return 0
    else:
        return 1


class Perceptron:
    def __init__(self, input_count, activation_func):
        self._weights = [random_float() for i in range(input_count)]
        self._threshold = abs(random_float())
        self._learningRate = 0.01
        self._activation_func = activation_func
        print("threshold is {}".format(self._threshold))

    def train(self, data, targets, max_times=1000):
        for i in range(max_times):
            print("{} times".format(i))
            error_total = 0
            for inputs, target in zip(data, targets):
                result = self.compute(inputs)
                if target != result:
                    error = target - result
                    error_total += abs(error)
                    self._update_weights(inputs, error)

            if error_total == 0:
                break

    def compute(self, inputs):
        result = round_float(self._sum_inputs(inputs) - self._threshold)
        y = self._activation_func(result)
        print("weights is {}, sum{} is {}, output is {}".format(self._weights, inputs, result, y))
        return y

    def _sum_inputs(self, inputs):
        return sum([x * w for x, w in zip(inputs, self._weights)])

    def _update_weights(self, inputs, e):
        self._weights = [round_float(w + self._learningRate * x * e) for x, w in zip(inputs, self._weights)]