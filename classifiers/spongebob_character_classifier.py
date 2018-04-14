# the CNN model to classify spongebob characters
import numpy as np
from models.data_model import DataModel
from services.data_preprocessor_service import DataPreprocessorService as dps


class SpongebobCharacterClassifier:

    x_train = []

    def __init__(self,
                 data: DataModel,
                 epochs: int,
                 layers: list
                 ):
        self.data = data
        self.epochs = epochs
        self.layers = layers

    # train model using this CNN architecture: X -> CONV -> RELU -> POOL -> FC -> SOFTMAX
    def train(self):
        # loop over layer objects calling their forward propogate methods
        for epoch in range(self.epochs):
            a_prev = self.data.x_train
            for i, layer in enumerate(self.layers):
                a_prev, cache = layer.forward_propogate(a_prev)
                self.layers[i] = layer  # layer has been updated with cache



