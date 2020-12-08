from abc import ABC
from itertools import tee

import numpy as np

# from itertools docs
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class Unit(ABC):

    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.forward(x)

class Network(Unit):
    r'''Represents a 3-layer MLP with configurable layer sizes.'''

    in_features: int
    out_features: int
    hidden_size: int
    model: [np.ndarray]

    def __init__(self, in_features: int, out_features: int, hidden_size: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.error = lambda z, t: 1/2 * np.square(np.linalg.norm(z - t))
        self.error_deriv = lambda z, t: -(z - t)

        self.model = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.model:
            y = layer(y)
        return y

    def backward_online(self, x: np.ndarray, t: np.ndarray, eta: float) -> np.ndarray:
        r'''A method for doing online backpropagation.'''

        # need the inputs and outputs from each layer
        y = x
        net = []
        out = []

        # each layer in this MLP is a linear layer followed by nonlinear activation
        for linear, nonlinear in self.model:
            y = linear(y)
            net.append(y)

            y = nonlinear(y)
            out.append(y)

        # compute output layer error
        delta = -eta * self.error_deriv(out[-1], t) * self.model[-1].derivative(net[-1])

        print(delta)
            

class Perceptron(Unit):
    r'''A light wrapper around a matrix to represent a perceptron.'''
    
    in_features: int
    out_features: int
    weight: np.ndarray
    bias: np.ndarray

    def __init__(self, in_features: int, out_features: int, weight: np.ndarray = None, bias: np.ndarray = None) -> None:
        r'''Initializes the perceptron weights (accounting for a bias term).'''
        self.in_features = in_features
        self.out_features = out_features

        self.initialize_parameters(weight, bias)

    def initialize_parameters(self, weight: np.ndarray, bias: np.ndarray) -> None:
        r'''Initializes the weights and bias parameters.'''

        if weight and weight.shape == (self.in_features, self.out_features):
            self.weight = weight
        else:
            raise ValueError('Initializing weights failed with invalid shape')

        if bias and bias.shape == (self.out_features, 1):
            self.bias = bias
        else:
            raise ValueError('Initializing bias failed with invalid shape')

        # random initialization, see 6.8.8 from text
        b = 1/np.sqrt(self.out_features)
        a = -1/np.sqrt(self.out_features)
        self.weight = (b - a) * np.random.random_sample((self.in_features, self.out_features)) + a
        self.bias = (b - a) * np.random.random_sample((self.out_features, 1)) + a

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight.T) + self.bias

class Sigmoid(Unit):
    r'''A class representing the Sigmoid activation function.'''

    def __init__(self) -> None:
        self.activation = lambda x: 1 / (1 + np.exp(-x))
        self.derivative = lambda x: self.activation(x) * (1 - self.activation(x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x)

class ReLU(Unit):
    r'''A class representing the ReLU activation function.'''

    def __init__(self) -> None:
        self.activation = lambda x: max(0, x)
        self.derivative = lambda x: 1 if x else 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x)

class Sgn(Unit):
    r'''A class representing the sign activation function.'''

    def __init__(self) -> None:
        self.activation = np.sign
        self.derivative = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x)