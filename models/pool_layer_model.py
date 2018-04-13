from models.pool_filter_model import PoolFilterModel


class PoolLayerModel:

    def __init__(self,
                 pool_filter: PoolFilterModel):
        self.pool_filter = pool_filter

