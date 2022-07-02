import numpy as np

def sig(x):# ｓ関数
    return 1 / (1 + np.exp(-x))

class singleNet:
    def __init__(self,weight,bias):
        self.weight = weight
        self.bias = bias

    def prog(self):
        g = np.dot(input, self.weight) + self.bias
        f = sig(g)
        return f

weight = np.array([0,1])
bias = 4

test_single = singleNet(weight,bias)
input = np.array([2,3])
test_single_f = test_single.prog()
print("test_single_f", test_single_f)
