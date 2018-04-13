# initialize epsilon for weights
import numpy as np


def epsilon_init(dimensions):
    ones = np.ones((1, len(dimensions)))
    e_init = np.sqrt(6) / np.square(ones.dot(dimensions))

    return int(e_init)
