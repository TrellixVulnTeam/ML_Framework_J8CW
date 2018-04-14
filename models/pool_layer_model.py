import numpy as np
from models.pool_filter_model import PoolFilterModel
from services.weight_initializer_service import WeightInitializerService


class PoolLayerModel:

    def __init__(self,
                 pool_filter: PoolFilterModel,
                 stride: int,
                 mode='max'):
        self.pool_filter = pool_filter
        self.stride = stride
        self.mode = mode
        self.weights = WeightInitializerService.random_initialize_filters([self.pool_filter.filter_size, self.pool_filter.filter_count])
        self.cache = {}

    def forward_propogate(self, A_prev):
        # get dims of previous input
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

        # get hparameters for this layer
        f = self.pool_filter.filter_size
        stride = self.stride

        # get dimensions of output volume
        n_H, n_W = self.compute_output_dimensions(n_H_prev, n_W_prev, f, stride)
        n_C = n_C_prev  # because with pool, channels don't expand or condense

        # initialize output matrix
        A = np.zeros((m, n_H, n_W, n_C))

        # loop over training examples
        for i in range(m):
            # select training example
            a_prev = A_prev[i]
            # loop over vertical axis of example
            for h in range(n_H):
                # loop over horizontal axis of example
                for w in range(n_W):
                    # find the corners of the slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # use corners to get slice of example to pool over
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                    # compute pool operation based on model's mode attra
                    if self.mode == 'max':
                        A[i, h, w, c] = a_prev_slice.max()
                    elif self.mode == 'average':
                        A[i, h, w, c] = a_prev_slice.mean()

        self.cache = {
            'A_prev': A_prev,
            'hparameters': {
                'filter_size': f,
                'stride': stride
            }
        }

        return A

    def compute_output_dimensions(self, n_H_prev: int, n_W_prev: int, filter_size: int, stride_size: int):
        n_H = int(1 + (n_H_prev - filter_size) / stride_size)
        n_W = int(1 + (n_W_prev - filter_size) / stride_size)

        return n_H, n_W
