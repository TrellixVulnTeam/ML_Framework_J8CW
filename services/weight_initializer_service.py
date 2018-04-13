# initializes the weights for the model to train
import numpy as np
import helpers.epsilon_initialize as ei

class WeightInitializerService:

    @staticmethod
    def random_initialize_filters(filter_dimentions, initializer):
        # open filter dimensions
        f_size, f_count = filter_dimentions

        # initialize epsilon <-- the range of random values
        e_init = ei.epsilon_init(filter_dimentions)

        # compute random weights
        w = np.random.rand(f_size, f_size, f_count).dot(2).dot(e_init).dot(e_init)
        b = np.random.rand(1, 1, 1).dot(2).dot(e_init).dot(e_init)

        return [w, b]
