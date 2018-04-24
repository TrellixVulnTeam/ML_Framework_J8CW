import numpy as np
from services.weight_initializer_service import DenseNNWeightInitializerService
from pathlib import Path


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
        if Path('stored/' + self.name + '_W').is_file() and Path('stored/' + self.name + '_b').is_file():
            W, b = np.loadtxt('stored/' + self.name + '_W'), np.loadtxt('stored/' + self.name + '_b')
            W = W.reshape(self.units_out, self.units_in)
            b = b.reshape(1, self.units_out)
        else:
            W, b = DenseNNWeightInitializerService.random_initialize_weights([self.units_in, self.units_out])
            W = W.reshape(self.units_out, self.units_in)
            b = b.reshape(1, self.units_out)

        self.W, self.b = W, b

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
        A_prev = self.forward_cache['A_prev']
        if len(A_prev.shape) > 2:
            m, n_H, n_W, n_C = self.forward_cache['A_prev'].shape
            A_prev = self.forward_cache['A_prev'].reshape(m, n_H * n_W * n_C)
        dW = (A_prev.T.dot(dZ)).T / self.m
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
        np.savetxt('stored/' + self.name + '_W', fW)
        np.savetxt('stored/' + self.name + '_b', fb)

        return self
