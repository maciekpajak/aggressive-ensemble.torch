from torchvision import transforms

from aggressive_ensemble.Tranforms.Transforms import *


class TransformImage(object):

    def __init__(self, input_size, mean, std, preprocessing=None, augmentations=None):

        self.input_size = input_size
        self.mean = mean
        self.std = std

        transform = []

        if preprocessing:
            preproc = []
            if preprocessing["polygon_extraction"]:
                preproc.append(ExtractPolygon())
            else:
                preproc.append(Crop())

            # preproc.append(Rescale(output_size=(self.input_size, self.input_size)))
            if preprocessing["ratation_to_horizontal"]:
                preproc.append(RotateToHorizontal())
            if preprocessing["edge_detection"]:
                preproc.append(EdgeDetection())
            if preprocessing["RGB_to_HSV"]:
                preproc.append(ChangeColorspace("RGB", 'HSV'))

            transform.extend(preproc)

        if augmentations:
            aug = []
            if augmentations["random_hflip"]:
                aug.append(RandomHorizontalFlip())
            if augmentations["random_vflip"]:
                aug.append(RandomVerticalFlip())
            if augmentations["random_rotation"]:
                aug.append(RandomRotate())
            if augmentations["change_RGB_channel"]:
                aug.append(SwitchRGBChannels())

            transform.extend(aug)

        transform.append(Rescale(output_size=(self.input_size, self.input_size)))
        transform.append(ToTensor())
        transform.append(Normalize(mean=self.mean, std=self.std))

        self.transform = transforms.Compose(transform)

    def __call__(self, sample):
        return self.transform(sample)
