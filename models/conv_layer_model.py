# the model object for a CONV Layer
import numpy as np
from models.conv_filter_model import CONVFilterModel
from services.weight_initializer_service import WeightInitializerService
import helpers.image_transform as it


class CONVLayerModel:

    def __init__(self,
                 conv_filter: CONVFilterModel,
                 stride: list,
                 padding: str):
        self.conv_filter = conv_filter
        self.stride = stride
        self.padding = padding
        self.cache = {}
        self.W, self.b = WeightInitializerService.random_initialize_filters([conv_filter.filter_size, conv_filter.filter_count])

    def forward_propogate(self, A_prev):
        # get dims from a and weight shapes
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, f, n_C_prev, n_C = self.W.shape

        # get height and width of output
        n_H, n_W = self.compute_output_dimensions(n_H_prev, n_W_prev, f)

        # initialize output volume as zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # pad a_prev with zeros
        pad = self.get_pad_size()
        stride = self.get_stride_size()
        A_prev_pad = it.zero_pad(A_prev, pad_size=pad, is_batch=True)

        # loop over the batch of training examples
        for i in range(m):
            # select ith training example
            a_prev_pad = A_prev_pad[i]
            # loop over the vertical axis of the training example
            for h in range(n_H):
                # loop over horizontal axis of training example
                for w in range(n_W):
                    # loop over classes/channels of training example
                    for c in range(n_C):
                        # find corneres of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # use these corners to get the slice to pass to forward_single_step()
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # convolve the 3D slice with the corresponding filter W and b (bias)
                        Z[i, h, w, c] = self.forward_single_step(a_slice_prev, self.W[:, :, :, c], self.b)

        # store info in "cache" to use in backprop
        self.cache = {
            'A_prev': A_prev,
            'W': self.W,
            'b': self.b,
            'hparameters': {
                'pad': pad,
                'stride': stride
            }
        }

        # return convolved output
        return Z

    def forward_single_step(self, a_prev_slice, W, b):
        s = np.multiply(a_prev_slice, W)
        Z = np.sum(s)
        Z += float(b)

        return Z

    def compute_output_dimensions(self, n_H_prev: int, n_W_prev:int, filter_size:int):
        pad_size = self.get_pad_size()
        stride_size = self.get_stride_size()
        filter_size = self.get_filter_size()
        n_H = int(((n_H_prev - filter_size + 2 * pad_size) / stride_size) + 1)
        n_W = int(((n_W_prev - filter_size + 2 * pad_size) / stride_size) + 1)

        return [n_H, n_W]

    def get_pad_size(self):
        pad = self.conv_filter.filter_size if self.padding == 'SAME' else 0
        return pad

    def get_stride_size(self):
        return self.stride[0]

    def get_filter_size(self):
        return self.conv_filter.filter_size
