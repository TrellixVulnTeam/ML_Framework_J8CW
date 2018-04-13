# service to preprocesses images
import helpers.image_transform as it


class DataPreprocessorService:

    @staticmethod
    def preprocess_imageset(imageset):
        for image in imageset:
            imageset[image] = it.square_crop_image(imageset[image])
            imageset[image] = it.resize_image(imageset[image], [64, 64])
            imageset[image] = it.zero_pad(imageset[image], pad_size=2, is_batch=False)

        return imageset
