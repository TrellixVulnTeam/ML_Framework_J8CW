# the CNN model to classify spongebob characters
import numpy as np


class SpongebobCharacterClassifier:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        # preprocess inputs (x)
        self.__preprocess_x_train()

    def __preprocess_x_train(self):
        x_train = self.x_train

        return True
