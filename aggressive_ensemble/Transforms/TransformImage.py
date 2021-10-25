from torchvision import transforms

from .Transforms import *


class TransformImage(object):
    """

    """
    transform = []

    def __init__(self, input_size: int, normalization: dict, preprocessing: list = None, augmentations: list = None):
        """

        Parameters
        ----------
        input_size :
        mean :
        std :
        preprocessing :
        augmentations :
        """
        input_size = input_size
        mean = normalization["mean"]
        std = normalization["std"]
        if preprocessing is None:
            preprocessing = []
        if augmentations is None:
            augmentations = []

        transform = preprocessing + augmentations

        transform.append(Rescale(output_size=(input_size, input_size)))
        transform.append(ToTensor())
        transform.append(Normalize(mean=mean, std=std))

        self.transformCompose = transforms.Compose(transform)

    def __call__(self, sample):
        return self.transformCompose(sample)

    def __str__(self):
        return "".join(str(x) + "->" for x in self.transform)
