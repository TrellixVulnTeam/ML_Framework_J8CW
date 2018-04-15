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
        self.prediction = None

    # train model using this CNN architecture: X (-> CONV -> RELU -> POOL) x 2 ... -> FC -> SOFTMAX
    def train(self):
        A_prev = np.zeros(self.data.x_train.shape)
        # loop over layer objects calling their forward propogate methods
        for epoch in range(self.epochs):
            # forward propogate and get predictions
            y_pred = self.forward_propogate()

            # compute the cost and use it in the backpropogation phase
            cost = self.compute_cost(y_pred)

            # use cost to perform backpropogations across the layers

            # update the weights


    def forward_propogate(self):
        A_prev = self.data.x_train
        for i, layer in enumerate(self.layers):
            A_prev = layer.forward_propogate(A_prev)
            self.layers[i] = layer  # layer has been updated with cache

        return A_prev

    def compute_cost(self, y_prediction):
        # todo: actual cost function
        return np.sum(self.data.y_train - y_prediction)

    def back_propogation(self, cost):
        return 0

    def update_weights(self):
        return 0

