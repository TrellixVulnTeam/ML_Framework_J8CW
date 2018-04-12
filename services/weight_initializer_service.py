# initializes the weights for the model to train
import numpy as np


class WeightInitializerService:

    @staticmethod
    def initialize_weights(dimensions, initializer):
        L_in, L_out = dimensions
        # initialize weights as zeros matrix
