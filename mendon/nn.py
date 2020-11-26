from abc import ABC
import numpy as np

class Unit(ABC):

    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        forward(x)

class Network(Unit):
    r'''Represents a 3-layer MLP with configurable layer sizes.'''

    in_features: int
    out_features: int
    hidden_size: int

    def __init__(self, in_features: int, out_features: int, hidden_size: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size

        model = [
            Perceptron((in_features, hidden_size)),
            Sigmoid(),
            Perceptron((hidden_size, out_features)),
            Sigmoid()
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in model:
            y = layer(y)
        return y

class Perceptron(Unit):
    r'''A light wrapper around a matrix to represent a perceptron.'''
    
    in_features: int
    out_features: int
    weight: np.ndarray
    bias: np.ndarray

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.ndarray((in_features, out_features))
        self.bias = np.ndarray((out_features))

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight.T) + self.bias

class Sigmoid(Unit):
    pass