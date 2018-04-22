import numpy as np
from services.weight_initializer_service import DenseNNWeightInitializerService


class FullyConnectedLayerModel:

    def __init__(self,
                 units_in: int,
                 units_out: int,
                 m: int,
                 name: str,
                 alpha: float = 1.0):
        self.units_in = units_in
        self.units_out = units_out
        self.m = m
        self.name = name
        self.alpha = alpha
        self.forward_cache = {}
        self.backward_cache = {}
        self.__load_weights()

    def __load_weights(self):
        weights = {
            'W': np.loadtxt('stored/' + self.name + '_W'),
            'b': np.loadtxt('stored/' + self.name + '_b')
        }
        if weights['W'] and weights['b']:
            self.W, self.b = weights['W'], weights['b']
        else:
            self.W, self.b = DenseNNWeightInitializerService.random_initialize_weights([self.units_in, self.units_out])

    def forward_propogate(self, A_prev):
        # get dims and use them to flatten A_prev
        m, n_H, n_W, n_C = A_prev.shape
        A_prev_reshaped = A_prev.reshape(m, n_H * n_W * n_C) if len(A_prev.shape) > 2 else A_prev

        a = A_prev_reshaped.dot(self.W.T)
        a += self.b

        self.forward_cache = {
            'A_prev': A_prev,
            'A': a,
            'W': self.W,
            'b': self.b
        }

        return a

    def backward_propogate(self, grads):
        dZ = grads['dZ']
        m, n_H, n_W, n_C = self.forward_cache['A_prev'].shape
        A_prev_reshaped = self.forward_cache['A_prev'].reshape(m, n_H * n_W * n_C) if len(self.forward_cache['A_prev'].shape) > 2 else self.forward_cache['A_prev']
        dW = (A_prev_reshaped.T.dot(dZ)).T / self.m
        db = np.sum(dZ) / self.m

        # update dZ for previous layer output
        dZ = self.W.T.dot(dZ.T)

        self.backward_cache = {
            'dZ': dZ,
            'dW': dW,
            'db': db
        }

        return {
            'dZ': dZ
        }

    def update_weights(self):
        self.W -= self.alpha * self.backward_cache['dW']
        self.b -= self.alpha * self.backward_cache['db']
        return self

    def store_weights(self):
        fW = self.W.flatten()
        fb = self.b.flatten()
        np.savetxt(self.name + '_W', fW)
        np.savetxt(self.name + '_b', fb)

        return self
