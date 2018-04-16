# the model for an activation layer
import numpy as np


class ActivationLayerModel:

    def __init__(self,
                 activation: str):
        self.activation = activation
        self.cache = {}

    def forward_propogate(self, A_prev):
        if self.activation == 'relu':
            A_activated = self.relu_activate(A_prev)
        elif self.activation == 'sigmoid':
            A_activated = self.sigmoid_activate(A_prev)
        elif self.activation == 'tanh':
            A_activated = self.tanh_activate(A_prev)
        elif self.activation == 'softmax':
            A_activated = self.softmax_activation(A_prev)
        else:
            A_activated = A_prev

        self.cache = {
            'A_prev': A_prev,
            'A': A_activated
        }

        return A_activated

    @staticmethod
    def relu_activate(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid_activate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh_activate(x):
        return np.tanh(x)

    @staticmethod
    def softmax_activation(x):
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum()
