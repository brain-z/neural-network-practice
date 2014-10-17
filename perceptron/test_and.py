"""
使用感知机处理逻辑"AND"问题
"""
__author__ = 'Kai'

import perceptron

inputData = [(0, 0), (0, 1), (1, 0), (1, 1)]
targetData = [0, 0, 0, 1]

print()
print("input:" + str(inputData))
print("target:" + str(targetData))

p = perceptron.Perceptron(2, perceptron.step_activation)
p.train(inputData, targetData)

print()

testData = [(0, 1), (1, 0), (1, 1), (0, 0), (1, 0), (0, 0), (1, 1)]
for x in testData:
    print("test " + str(x) + " result is " + str(p.compute(x)))


