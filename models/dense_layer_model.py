# the model for a dense layer
import numpy as np


class DenseLayerModel:

    def __init__(self,
                 units: int,
                 activation: str,
                 weights: list,
                 bias: int):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.bias = bias
