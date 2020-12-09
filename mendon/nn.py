from abc import ABC
from itertools import tee

import numpy as np  # type: ignore

class Unit(ABC):

    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Network(Unit):
    r'''Represents a 3-layer MLP with configurable layer sizes.'''

    in_features: int
    out_features: int
    hidden_size: int
    model: list

    def __init__(self, in_features: int, hidden_size: int, out_features: int, weight = dict()) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.error = lambda z, t: 1/2 * np.square(np.linalg.norm(t - z))
        self.error_deriv = lambda z, t: -(t - z)

        self.model = [
            Perceptron(in_features, hidden_size),
            ReLU(),
            Perceptron(hidden_size, out_features),
            Sigmoid(),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.model:
            y = layer(y)
        return y

    def backward_online(self, x: np.ndarray, t: np.ndarray, eta: float) -> None:
        r'''Peform single-sample backpropagation.'''

        # need the inputs and outputs from each layer
        net_j = self.model[0](x)
        y = self.model[1](net_j)
        net_k = self.model[2](y)
        z = self.model[3](net_k)

        # use the results to backpropagate
        sens_k = self.error_deriv(z, t) * self.model[3].derivative(net_k)
        sens_j = np.matmul(self.model[2].weight, sens_k) * self.model[1].derivative(net_j)

        # update
        self.model[2].weight = self.model[2].weight - eta * np.matmul(y, sens_k.T)
        self.model[2].bias = self.model[2].bias - eta * self.model[2].bias * sens_k
    
        self.model[0].weight = self.model[0].weight - eta * np.matmul(x, sens_j.T)
        self.model[0].bias = self.model[0].bias - eta * self.model[0].bias * sens_j

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
        elif weight:
            raise ValueError('Initializing weights failed with invalid shape')

        if bias and bias.shape == (self.out_features, 1):
            self.bias = bias
        elif bias:
            raise ValueError('Initializing bias failed with invalid shape')

        # random initialization, see 6.8.8 from text
        b = 1/np.sqrt(self.in_features)
        a = -1/np.sqrt(self.in_features)
        self.weight = (b - a) * np.random.random_sample((self.in_features, self.out_features)) + a
        self.bias = (b - a) * np.random.random_sample((self.out_features, 1)) + a

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(self.weight.T, x) + self.bias

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
        self.activation = lambda x: np.where(x < 0, 0, x)
        self.derivative = lambda x: np.where(x <= 0, 0, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x)

class Sgn(Unit):
    r'''A class representing the sign activation function.'''

    def __init__(self) -> None:
        self.activation = np.sign
        self.derivative = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x)