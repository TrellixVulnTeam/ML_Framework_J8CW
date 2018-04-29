# the CNN model to classify spongebob characters
import numpy as np
from models.data_model import DataModel
import matplotlib.pyplot as plt
from services.gradient_check_service import GradientCheckService


class SpongebobCharacterClassifier:

    x_train = []

    def __init__(self,
                 data: DataModel,
                 epochs: int,
                 layers: list,
                 gradient_check: bool = False
                 ):
        self.data = data
        self.epochs = epochs
        self.layers = layers
        self.prediction = None
        self.cost_history = []
        self.y_pred = []
        self.gradient_check = gradient_check

    # train model using this CNN architecture: X -> CONV -> POOL -> FC -> SOFTMAX
    def train(self, x, y):
        # self.display_data(x, y)
        # loop over epochs and perform gradient descent
        for epoch in range(self.epochs):
            # print('Epoch: ' + str(epoch))

            # forward propogate and get predictions
            self.y_pred = self.forward_propogate(x)

            # compute the cost and use it to track J_history
            cost = self.compute_cost(y, self.y_pred)

            # print('Cost: ' + str(cost))
            self.cost_history.append(cost)

            # use cost to perform backpropogations across the layers
            self.backward_propogate(y)

            GradientCheckService.check_gradients(self.layers[0], self) if self.gradient_check else None

            # update the weights
            self.update_weights(epoch + 1)  # plus 1 to avoid divide by zero in momentum

    def forward_propogate(self, A_prev):
        for layer in self.layers:
            A_prev = layer.forward_propogate(A_prev)

        return A_prev

    def compute_cost(self, y, y_prediction):
        m = y.shape[0]
        cost = -(np.sum(y * np.log(y_prediction + 0.001) + (1 - y) * np.log(1 - y_prediction + 0.00000001))) / m  # added + 0.00000001 to avoid log of zeros
        return cost

    def backward_propogate(self, y):
        # get starting grad for y prediction
        dZ = np.subtract(self.y_pred, y)

        grads = {
            'dZ': dZ
        }

        # add grads to skipped layer
        self.layers[len(self.layers) - 1].backward_cache = grads

        for layer in reversed(self.layers[:-1]):  # skip output layer as it is computed above
            grads = layer.backward_propogate(grads)

        return grads

    def update_weights(self, iteration: int):
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.update_weights(iteration)

        return True

    def store_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.store_weights()

    @staticmethod
    def display_data(x, y):
        for i, image in enumerate(x):
            plt.imshow(image)
            plt.title(np.argmax(y[i]))
            plt.show()
