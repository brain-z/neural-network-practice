__author__ = 'Kai'
from multilayer_perceptron import MultilayerPerceptron

network = MultilayerPerceptron([2, 2, 1], 10, constant_factor=0.00095, learning_rate_increase=0.1, learning_rate_decrease=0.08)
converge = network.train([[1, 0], [0, 0], [0, 1], [1, 1]], [[1], [0], [1], [0]], 200)

print("%s times learning."%network.learning_times)
print("="*20)
if converge != True:
    print("  !!!!!!network not converge!!!!!!")
    print()

print("  thresholds:%s"%network.thresholds)
print("  weights:%s"%network.weights)
print()
print("  XOR[1, 0] ->", round(network.compute([1, 0])[0], 0))
print("  XOR[0, 0] ->", round(network.compute([0, 0])[0], 0))
print("  XOR[0, 1] ->", round(network.compute([0, 1])[0], 0))
print("  XOR[1, 1] ->", round(network.compute([1, 1])[0], 0))

