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

weight = np.array([1,1])
bias = 0

test_single = singleNet(weight,bias)
input = np.array([-2,-1])
test_single_f = test_single.prog()
print("test_single_f", test_single_f)

# ----------------------------------------------------------

class doubleNet:
    def h_out(self):
        h1g = singleNet(weight,bias)
        h2g = singleNet(weight,bias)

        h1 = h1g.prog()
        h2 = h2g.prog()

        return np.array([h1,h2])

test_double = singleNet(weight,bias)
h = doubleNet()
h_out = h.h_out()
input = h_out
test_double_f = test_double.prog()
print("test_double_f", test_double_f)