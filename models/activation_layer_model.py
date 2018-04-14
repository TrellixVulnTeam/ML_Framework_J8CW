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
        elif self.activation == 'softmax':
            self.softmax_activation(A_prev)

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
