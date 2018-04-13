# the model object for a CONV Layer
from models.conv_filter_model import CONVFilterModel
from services.weight_initializer_service import WeightInitializerService

class CONVLayerModel:

    def __init__(self,
                 conv_filter: CONVFilterModel,
                 stride: list,
                 padding: str):
        self.conv_filter = conv_filter
        self.stride = stride
        self.padding = padding
        self.weights = WeightInitializerService.random_initialize_filters([conv_filter.filter_size, conv_filter.filter_count])

    def forward_propogate(self, a_prev: list):
        a = []
        return a
