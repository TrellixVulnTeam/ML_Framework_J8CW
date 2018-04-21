# model structure for the data to pass to classifier
import numpy as np
from services.data_preprocessor_service import DataPreprocessorService as dps


class DataModel:

    def __init__(self, data: dict, num_classes: int, image_size: list):
        self.x = dps.preprocess_imageset(data['x'], image_size)
        self.y = dps.one_hot_encode(data['y'], num_classes)
