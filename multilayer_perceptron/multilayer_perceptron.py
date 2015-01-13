__author__ = 'Kai'

import random
import math
import time
import decimal
import copy

class MultilayerPerceptron:
    def __init__(self, sizes, learning_rate, min_error=0.001, constant_factor=0, learning_rate_increase=1.05, learning_rate_decrease=0.9):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.learning_rate_increase = learning_rate_increase
        self.learning_rate_decrease = learning_rate_decrease
        self.min_error = min_error
        self.learning_times = 0
        self.constant_factor = constant_factor
        self._weights_log = None

        # 初始化权重
        self.weights = [[[self._random_val() for z in range(sizes[i + 1])]  for j in range(sizes[i])] for i in range(len(sizes) - 1)]
        self._weights_log = [[[0 for z in range(sizes[i + 1])]  for j in range(sizes[i])] for i in range(len(sizes) - 1)]
        # 初始化阈值
        self.thresholds = [[self._random_val() for j in range(sizes[i])] for i in range(1, len(sizes))]
        self._thresholds_log = [[0 for j in range(sizes[i])] for i in range(1, len(sizes))]

    def train(self, inputs, targets, max_times=100):
        converge = False
        error_square_sum_log = 0;
        for i in range(max_times):
            self.learning_times = i + 1
            print("No.%s"%(i + 1), "times learning")
            print("=" * 20)
            pass_total = 0
            for p in range(len(inputs)):
                print("  No.%s"%(p + 1), "specimen")
                print("  learning rate:%s"%self.learning_rate)
                print("  thresholds:%s"%self.thresholds)
                print("  weigths:%s"%self.weights)
                outputs = self._compute_outputs(inputs[p])
                print("  outputs:%s"%outputs)
                errors = self._compute_errors(targets[p], outputs[-1])
                error_square_sum = self._error_square_sum(sum(errors))

                if error_square_sum > error_square_sum_log * 1.04:
                    self.learning_rate -= self.learning_rate * self.learning_rate_decrease
                else:
                    self.learning_rate += self.learning_rate * self.learning_rate_increase

                error_square_sum_log = error_square_sum

                print("  input:%s"%inputs[p], "target:%s"%targets[p] , "output:%s"%outputs[-1], "error:%s"%errors, "square sum:%s"%error_square_sum)
                if error_square_sum <= self.min_error:
                    pass_total += 1
                else:
                    self._update(outputs, errors)
                print()
                self._weights_log = copy.deepcopy(self.weights)

            if pass_total == len(inputs):
                converge = True
                break

        return converge

    def compute(self, input):
        return self._compute_outputs(input)[-1]

    def _compute_outputs(self, input):
        n = len(self.sizes)
        outputs = [[0] * self.sizes[i] for i in range(n)]
        for i in range(n):
            if i == 0:
                outputs[i] = input
            else:
                outputs[i] = self._output(outputs[i - 1], self.weights[i - 1], self.thresholds[i - 1])

        return outputs

    def _output(self, input, weight, threshold):
        return [self._activate(input, [weight[j][k] for j in range(len(input))], threshold[k]) for k in range(len(threshold))]

    def _activate(self, input, weight, threshold):
        return self._sigmoid_func(sum([x * w for x, w in zip(input, weight)]) - threshold)

    def _compute_errors(self, target, output):
        return [yd - y for yd, y in zip(target, output)]

    def _update(self, outputs, errors):
        n = len(self.sizes)
        slopes = [[0] * self.sizes[i] for i in range(n)]
        for i in reversed(range(1, n)):
            if i < n - 1:
                _weights = self.weights[i]
                _slopes = slopes[i + 1]
                errors = [sum([_slopes[k] * _weights[j][k] for k in range(self.sizes[i + 1])]) for j in range(self.sizes[i])]

            slopes[i] = [y * (1 - y) * e for y, e in zip(outputs[i], errors)]

        r = self.learning_rate
        for i in range(1, n):
            _weights_log = self._weights_log[i - 1]
            _weights = self.weights[i - 1]
            _input = outputs[i - 1]
            _slopes = slopes[i]
            self.weights[i - 1] = [[self.constant_factor * _weights_log[j][k] +  _weights[j][k] + (r * _input[j] * _slopes[k]) for k in range(self.sizes[i])] for j in range(self.sizes[i - 1])]

        for i in range(1, n):
            _thresholds = self.thresholds[i - 1]
            _slopes = slopes[i]
            self.thresholds[i - 1] = [_thresholds[j] + (r * -1 * _slopes[j]) for j in range(len(_thresholds))]

    def _random_val(self):
        val = 2.4 / self.sizes[0]
        return random.uniform(-val, val)

    @staticmethod
    def _sigmoid_func(val):
        return 1 / (1 + math.exp(-val))

    @staticmethod
    def _error_square_sum(e):
        return e ** 2 + e ** 2