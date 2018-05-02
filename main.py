from classifiers.spongebob_character_classifier import SpongebobCharacterClassifier
from services.data_preprocessor_service import DataPreprocessorService as dps
from models.data_model import DataModel
from services.layer_initializer_service import LayerInitializerService
from os import listdir
from skimage.data import imread
from helpers.prediction_helper import PredictionHelper
import matplotlib.pyplot as plt
from helpers.class_to_character_helper import ClassToCharacterHelper
import numpy as np


directory = 'datasets/user_images'
if listdir(directory):
    print("Setting up classifier...")
    data = dps.load_data()
    data_model = DataModel(data, 7, [150, 150])
    layers = LayerInitializerService.load_layers(7, 0.01)
    classifier = SpongebobCharacterClassifier(data_model, 1000, layers, 5)

    print("Training classifier... this may take a couple minutes")
    classifier.train(classifier.data.x_train, classifier.data.y_train)
    print("Training complete!")

    print("..................")

    print("Testing on your image...")
    image_name = listdir(directory)[0]
    image = imread(directory + '/' + image_name)
    image = dps.preprocess_imageset([image], [150, 150])[0]
    inputs = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]))
    inputs[0, :, :, :] = image
    pred = classifier.forward_propogate(inputs)

    print("Making prediction!")
    class_prediction = PredictionHelper.predict(pred)[0]
    character_name = ClassToCharacterHelper.character_map[class_prediction]
    plt.title(character_name)
    plt.imshow(image)
    plt.show()
else:
    print("Please save a character image to /datasets/user_images!")
