import numpy as np
from services.weight_initializer_service import DenseNNWeightInitializerService


class FullyConnectedLayerModel:

    def __init__(self,
                 units_in: int,
                 units_out: int):
        self.units_in = units_in
        self.units_out = units_out
        self.W, self.b = DenseNNWeightInitializerService.random_initialize_weights([self.units_in, self.units_out])

    def forward_propogate(self, A_prev):
        # get dims and use them to flatten A_prev
        m, n_H, n_W, n_C = A_prev.shape
        A_prev_reshaped = A_prev.reshape(m, n_H * n_W * n_C)

        a = A_prev_reshaped.dot(self.W.T)
        a += self.b

        return a
