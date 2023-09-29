import numpy as np
import math

class Layer:
    def __init__(self, n_inputs):
        self.size = n_inputs
    def forward(self, inputs):
        self.output = []
        for i in range(self.size):
            self.output.append([inputs[i]])
        self.output = np.array(self.output)
        return self.output

    def backward(self, input, learning_rate):
        pass

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons, 1)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights.T, self.inputs) + self.biases
        return self.output

    def backward(self, input, learning_rate=.000001):
        weights_grad = np.dot(self.inputs, input.T)
        self.weights = self.weights - learning_rate * weights_grad
        self.biases = self.biases - learning_rate * input
        return np.dot(self.weights, input)


class Activation:
    def __init__(self):
        pass

    def sigmoid(self, inputs, back=0):
        self.inputs = inputs
        self.output = 1/(1+math.e**-self.inputs)
        self.gradient = self.output * (1 - self.output)
        if back == 1:
            return self.gradient
        else:
            return self.output

    def ReLU(self, inputs, back=0):
        self.inputs = inputs
        for i in range(len(self.inputs)):
            for j in range(len(self.inputs[i])):
                if self.inputs[i][j] <= 0:
                    self.inputs[i][j] = 0
        self.gradient = self.inputs
        for i in range(len(self.gradient)):
            for j in range(len(self.gradient[i])):
                if self.gradient[i][j] != 0:
                    self.gradient[i][j] = 1
        if back == 1:
            return self.gradient
        else:
            return self.inputs

    def softmax(self, inputs, back=0):
        self.inputs = inputs
        denom = 0
        for i in range(len(self.inputs)):
            denom += math.e**self.inputs[i]
        self.output = (math.e**self.inputs)/denom
        self.gradient = ((math.e**self.inputs)/denom) * (1 - (math.e**self.inputs)/denom)
        if back == 1:
            return self.gradient
        else:
            return self.output


class Loss:
    def __init__(self, y_pred, y):
        self.inputs = y_pred.flatten()
        self.expected = y

    def MSE(self, back=0):
        self.output = (1/len(self.inputs))*((self.expected - self.inputs)**2).sum()
        self.gradient = np.dot(self.expected, self.inputs)
        if back == 1:
            return self.gradient
        else:
            return self.output

    def cross_entropy(self, back=0):
        self.output = -((self.expected * np.log(self.inputs)).sum())
        self.gradient = -(self.expected/self.inputs).sum()
        if back == 1:
            return self.gradient
        else:
            return self.output