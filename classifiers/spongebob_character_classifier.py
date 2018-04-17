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

            # compute the cost and use it in the backpropogation phase
            cost = self.compute_cost(self.y_pred)

            # use cost to perform backpropogations across the layers
            self.back_propogation(cost)

            # update the weights


    def forward_propogate(self):
        A_prev = self.data.x
        for i, layer in enumerate(self.layers):
            A_prev = layer.forward_propogate(A_prev)
            self.layers[i] = layer  # layer has been updated with cache

        return A_prev

    def compute_cost(self, y_prediction):
        m = self.data.y.shape[0]
        cost = -(np.sum(self.data.y * np.log(y_prediction) + (1 - self.data.y) * np.log(1 - y_prediction))) / m
        return cost

    def back_propogation(self, cost):
        grad = cost
        for layer in reversed(self.layers):
            grad = layer.backwards_propogate(grad)

        return grad

    def update_weights(self):
        return 0

