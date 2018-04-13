# the CNN model to classify spongebob characters
import numpy as np
from models.data_model import DataModel
from services.data_preprocessor_service import DataPreprocessorService as dps


class SpongebobCharacterClassifier:

    x_train = []

    def __init__(self,
                 data: DataModel,

                 ):
        self.data = data

    # train model using this CNN architecture: X -> CONV -> RELU -> POOL -> FC -> SOFTMAX
    def train(self):
        # preprocess inputs (x)
        self.x_train = dps.preprocess_imageset(self.data.x_train)

        # initialize layer objects
        self

    def initialize_layer_objects(self):
        