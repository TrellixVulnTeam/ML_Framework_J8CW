# the CNN model to classify spongebob characters
import numpy as np
from models.data_model import DataModel
from services.data_preprocessor_service import DataPreprocessorService as dps
from helpers.prediction_helper import PredictionHelper


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
        self.prediction = None
        self.y_pred = []

    # train model using this CNN architecture: X (-> CONV -> RELU -> POOL) x 2 ... -> FC -> SOFTMAX
    def train(self):
        A_prev = np.zeros(self.data.x.shape)
        # loop over layer objects calling their forward propogate methods
        for epoch in range(self.epochs):
            # forward propogate and get predictions
            Z = self.forward_propogate()

            # make prediction
            self.y_pred = PredictionHelper.predict(Z)

            # compute the cost and use it in the backpropogation phase
            # cost = self.compute_cost(y_pred)

            # use cost to perform backpropogations across the layers

            # update the weights


    def forward_propogate(self):
        A_prev = self.data.x
        for i, layer in enumerate(self.layers):
            A_prev = layer.forward_propogate(A_prev)
            self.layers[i] = layer  # layer has been updated with cache

        return A_prev

    def compute_cost(self, y_prediction):
        # todo: actual cost function
        return np.sum(self.data.y - y_prediction)

    def back_propogation(self, cost):
        return 0

    def update_weights(self):
        return 0

