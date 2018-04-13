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


    def forward_propogate(self, a_prev: list):
        a = []
        return a
