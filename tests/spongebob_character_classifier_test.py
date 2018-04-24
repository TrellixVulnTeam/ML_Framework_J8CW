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
        self.data_model = data_model.DataModel(shuffled_imageset, max(imageset['y']) + 1, [64, 64])

    def run(self):
        # conv layer 1
        conv_filter_1 = conv_filter_model.CONVFilterModel(4, 3, 8)
        conv_layer_1 = conv_layer_model.CONVLayerModel(conv_filter_1, [2, 2], 'same', 'conv1', 0.001)

        # pool layer 1
        pool_filter_1 = pool_filter_model.PoolFilterModel(4, 8)
        pool_layer_1 = pool_layer_model.PoolLayerModel(pool_filter_1, 2, 'max')

        # fully connected layer
        fc_layer = fully_connected_layer_model.FullyConnectedLayerModel(1568, self.num_classes, len(self.data_model.y), 'fc1', 0.001)

        # final activation layer
        activation_layer = activation_layer_model.ActivationLayerModel('softmax', 'output_activation')

        # layers list
        layers = [
            conv_layer_1,
            pool_layer_1,
            fc_layer,
            activation_layer
        ]

        # instantiate classifier model
        classifier_model = SpongebobCharacterClassifier(self.data_model, 1000, layers)

        # train model
        classifier_model.train()

        return classifier_model
