# the model for an activation layer
import numpy as np


class ActivationLayerModel:

    def __init__(self,
                 activation: str):
        self.activation = activation

    def forward_propogate(self, A_prev):
        if self.activation == 'relu':
            self.relu_activate(A_prev)
        elif self.activation == 'sigmoid':
            self.sigmoid_activate(A_prev)
        elif self.activation == 'tanh':
            self.tanh_activate(A_prev)

    def relu_activate(self, x):
        return np.maximum(x, 0)

    def sigmoid_activate(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh_activate(self, x):
        return np.tanh(x)
