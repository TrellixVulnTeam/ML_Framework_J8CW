# the test process for the spongebob character classifier

from services.data_preprocessor_service import DataPreprocessorService
from classifiers.spongebob_character_classifier import SpongebobCharacterClassifier
from models import *


class SpongebobCharacterClassifierTest:

    def __init__(self, num_classes: int, training_phase: str = 'train'):
        self.num_classes = num_classes
        imagesets = DataPreprocessorService.load_imagesets(training_phase)
        imageset = DataPreprocessorService.merge_imagesets(imagesets)
        shuffled_imageset = DataPreprocessorService.unison_shuffle_images_labels(imageset['x'], imageset['y'])
        self.data_model = data_model.DataModel(shuffled_imageset, max(imageset['y']) + 1, [100, 100])
        self.learning_rate = 0.01

    def run(self):
        fc_layer_1 = fully_connected_layer_model.FullyConnectedLayerModel(30000, 10, len(self.data_model.y), 'fc1', self.learning_rate)
        activation_layer_1 = activation_layer_model.ActivationLayerModel('relu', 'output_activation')
        output_fc = fully_connected_layer_model.FullyConnectedLayerModel(10, self.num_classes, len(self.data_model.y), 'fc2', self.learning_rate)
        output_activation = activation_layer_model.ActivationLayerModel('softmax', 'output_activation')

        # layers list
        layers = [
            fc_layer_1,
            activation_layer_1,
            output_fc,
            output_activation
        ]

        # instantiate classifier model
        classifier_model = SpongebobCharacterClassifier(self.data_model, 1000, layers)

        # train model
        classifier_model.train()

        return classifier_model
