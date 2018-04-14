# model structure for the data to pass to classifier
from services.data_preprocessor_service import DataPreprocessorService as dps


class DataModel:

    def __init__(self, data: dict, pad: int, image_size: list):
        self.x_train = dps.preprocess_imageset(data['x_train'], image_size, pad)
        self.y_train = dps.preprocess_imageset(data['y_train'], image_size, pad)
        self.x_val = dps.preprocess_imageset(data['x_val'], image_size, pad)
        self.y_val = dps.preprocess_imageset(data['y_val'], image_size, pad)
        self.x_test = dps.preprocess_imageset(data['x_test'], image_size, pad)
        self.y_test = dps.preprocess_imageset(data['y_test'], image_size, pad)
