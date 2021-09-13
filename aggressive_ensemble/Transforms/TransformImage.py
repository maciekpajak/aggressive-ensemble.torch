from torchvision import transforms

from aggressive_ensemble.Transforms.Transforms import *


class TransformImage(object):
    """

    """
    transform = []

    def __init__(self, input_size, mean, std, preprocessing=None, augmentations=None):
        """

        Parameters
        ----------
        input_size :
        mean :
        std :
        preprocessing :
        augmentations :
        """
        self.input_size = input_size
        self.mean = mean
        self.std = std

        self.transform = []
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

            self.transform.extend(preproc)

        if augmentations:
            aug = []
            if augmentations["random_hflip"]:
                aug.append(RandomHorizontalFlip())
            if augmentations["random_vflip"]:
                aug.append(RandomVerticalFlip())
            if augmentations["random_rotation"]:
                aug.append(RandomRotate())
            if augmentations["switch_RGB_channel"]:
                aug.append(SwitchRGBChannels())

            self.transform.extend(aug)

        self.transform.append(Rescale(output_size=(self.input_size, self.input_size)))
        self.transform.append(ToTensor())
        self.transform.append(Normalize(mean=self.mean, std=self.std))

        self.transformCompose = transforms.Compose(self.transform)

    def __call__(self, sample):
        return self.transformCompose(sample)

    def __str__(self):
        return "".join(str(x) + "->" for x in self.transform)
