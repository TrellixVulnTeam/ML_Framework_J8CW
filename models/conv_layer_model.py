# the model object for a CONV Layer
from models.conv_filter_model import CONVFilterModel


class CONVLayerModel:

    def __init__(self,
                 conv_filter: CONVFilterModel,
                 stride: list,
                 padding: str):
        self.conv_filter = conv_filter
        self.stride = stride
        self.padding = padding

    def forward_propogate(self, inputs: list):
        outputs = []
        return outputs
