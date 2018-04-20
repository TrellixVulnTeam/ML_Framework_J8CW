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
        self.cost_history = []
        self.y_pred = []

    # train model using this CNN architecture: X (-> CONV -> RELU -> POOL) x 2 ... -> FC -> SOFTMAX
    def train(self):
        # loop over epochs and perform gradient descent
        for epoch in range(self.epochs):
            # forward propogate and get predictions
            self.y_pred = self.forward_propogate()

            # compute the cost and use it to track J_history
            cost = self.compute_cost(self.y_pred)
            print(cost)
            self.cost_history.append(cost)

            # use cost to perform backpropogations across the layers
            self.backward_propogate()

            # update the weights
            self.update_weights()

    def forward_propogate(self):
        A_prev = self.data.x
        for i, layer in enumerate(self.layers):
            A_prev = layer.forward_propogate(A_prev)

        return A_prev

    def compute_cost(self, y_prediction):
        m = self.data.y.shape[0]
        cost = -(np.sum(self.data.y * np.log(y_prediction) + (1 - self.data.y) * np.log(1 - y_prediction))) / m
        return cost

    def backward_propogate(self):
        # get starting grad for y prediction
        dZ = np.subtract(self.y_pred, self.data.y)

        grads = {
            'dZ': dZ
        }

        # add grads to skipped layer
        self.layers[len(self.layers) - 1].backward_cache = grads

        for i, layer in enumerate(reversed(self.layers[:-1])):  # skip output layer as it is computed above
            grads = layer.backward_propogate(grads)

        return grads

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            layer.update_weights()

        return True
