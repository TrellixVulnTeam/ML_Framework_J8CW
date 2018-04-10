

class ImagePreprocessor:

    def __init__(self, image):
        self.image = image

    def crop_image(self):
        image_shape = self.image.shape
        [long_side, short_side] = [image_shape[0], image_shape[1]] if image_shape[0] > image_shape[1] else [image_shape[1], image_shape[0]]
        indent_size = (long_side - short_side) / 2
        cropped_image = self.image[indent_size:(long_side - indent_size), :, :] if long_side == image_shape[0] else self.image[:, indent_size:(long_side - indent_size), :]

        return cropped_image

    def pad_image(self, pad_type, pad_size):

        return

    def resize_image(self, desired_size):

        return
