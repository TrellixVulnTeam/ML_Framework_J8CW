import numpy as np
from services.weight_initializer_service import DenseNNWeightInitializerService
from models.activation_layer_model import ActivationLayerModel

class FullyConnectedLayerModel:

    def __init__(self,
                 units_in: int,
                 units_out: int,
                 m: int):
        self.units_in = units_in
        self.units_out = units_out
        self.m = m
        self.W, self.b = DenseNNWeightInitializerService.random_initialize_weights([self.units_in, self.units_out])
        self.forward_cache = {}
        self.backward_cache = {}

    def forward_propogate(self, A_prev):
        # get dims and use them to flatten A_prev
        m, n_H, n_W, n_C = A_prev.shape
        A_prev_reshaped = A_prev.reshape(m, n_H * n_W * n_C)

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
        dW = (self.forward_cache['A_prev'].T.dot(dZ)) / self.m
        db = np.sum(dZ, axis=1, keepdims=True) / self.m

        # update dZ for previous layer output
        dZ = self.W.T.dot(dZ)

        self.backward_cache = {
            'dZ': dZ,
            'dW': dW,
            'db': db
        }

        return dZ

    def update_weights(self):
        return True
