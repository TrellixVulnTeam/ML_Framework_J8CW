# service to preprocesses images
import helpers.image_transform as it
from skimage.data import imread
import sklearn.utils as utils
from os import listdir
import numpy as np


class DataPreprocessorService:

    @staticmethod
    def load_imagesets(data_type: str):
        imagesets = []
        directory = 'datasets/' + data_type
        for c, imageset in enumerate(listdir(directory)):
            c_images = []
            c_labels = []
            directory += '/' + imageset
            for image in listdir(directory):
                c_images.append(imread(directory + '/' + image))
                c_labels.append(c)

            c_image_data = {
                'images': c_images,
                'labels': c_labels
            }

            imagesets.append(c_image_data)

            # reset directory
            directory = 'datasets/train'

        return imagesets

    @staticmethod
    def merge_imagesets(imagesets: list):
        merged_images = []
        merged_labels = []
        for imageset in imagesets:
            merged_images += imageset['images']
            merged_labels += imageset['labels']

        return {
            'images': merged_images,
            'labels': merged_labels
        }

    @staticmethod
    def preprocess_imageset(imageset, image_size: list, pad):
        for image in imageset:
            imageset[image] = it.square_crop_image(imageset[image])
            imageset[image] = it.resize_image(imageset[image], image_size)

        return imageset

    @staticmethod
    def unison_shuffle_images_labels(images: list, labels: list):
        results = utils.shuffle(images, labels, random_state=np.random.randint(10))

        return {
            'images': results[0],
            'labels': results[1]
        }

    @staticmethod
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels]
