# service to preprocesses images
import helpers.image_transform as it


class DataPreprocessorService:

    @staticmethod
    def preprocess_imageset(imageset, image_size: list, pad):
        for image in imageset:
            imageset[image] = it.square_crop_image(imageset[image])
            imageset[image] = it.resize_image(imageset[image], image_size)

        return imageset
