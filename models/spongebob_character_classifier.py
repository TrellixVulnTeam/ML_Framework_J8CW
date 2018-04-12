# the CNN model to classify spongebob characters
import numpy as np
from services.data_preprocessor_service import DataPreprocessorService as dps


class SpongebobCharacterClassifier:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # train model using this CNN architecture: X -> CONV -> RELU -> POOL -> FC -> SOFTMAX
    def train(self):
        # preprocess inputs (x)
        self.x_train = dps.preprocess_imageset(self.x_train)

