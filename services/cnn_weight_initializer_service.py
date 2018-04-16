# initializes the weights for the model to train
import numpy as np
import helpers.epsilon_initialize as ei


class CNNWeightInitializerService:

    @staticmethod
    def random_initialize_filters(filter_dimentions: list):
        # open filter dimensions
        f_size, channels_in, channels_out = filter_dimentions

        # initialize epsilon <-- the range of random values
        e_init = ei.epsilon_init(filter_dimentions)

        # compute random weights
        W = np.random.rand(f_size, f_size, channels_in, channels_out).dot(2).dot(e_init).dot(e_init)
        b = np.random.rand(1, 1, 1, channels_out).dot(2).dot(e_init).dot(e_init)

        return [W, b]
