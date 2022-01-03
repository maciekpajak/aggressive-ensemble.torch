from typing import Tuple

import torch
import numpy as np
from torchvision import transforms
from imgaug.augmentables.polys import Polygon
from imgaug import augmenters as iaa

from src.aggressive_ensemble.utils.transform import transform

__all__ = ['Rescale', 'RotateToHorizontal', 'EdgeDetection', 'ChangeColorspace', 'ExtractPolygon', 'Crop',
           'Normalize', 'ToTensor']


class RotateToHorizontal(transform):
    """Transformacja obracająca obiekt na obrazie do pozycji horyzontalnej"""

    def __call__(self, image, polygon, labels):
        """

        :param sample:
        :type sample:
        :return:
        :rtype:
        """

        h, w = image.shape[:2]
        y1, x1 = polygon[1, 1] - polygon[0, 1], polygon[1, 0] - polygon[0, 0]
        y2, x2 = polygon[2, 1] - polygon[1, 1], polygon[2, 0] - polygon[1, 0]
        object_w = max((np.sqrt(x1 * x1 + y1 * y1)), (np.sqrt(x2 * x2 + y2 * y2)))

        if (x1 * x1 + y1 * y1) > (x2 * x2 + y2 * y2):
            angle = np.arctan2(y1, x1) * 180 / np.pi
        else:
            angle = np.arctan2(y2, x2) * 180 / np.pi

        pad = max(0, int(10 - (w - object_w) / 2))
        t = iaa.Sequential([iaa.Pad(px=pad), iaa.Affine(rotate=-angle)])
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Rotiation to horizontal"


class EdgeDetection(transform):
    """Transformacja przeprowadzająca detekcję krawędzi"""

    def __call__(self, image, polygon, labels):
        t = iaa.EdgeDetect(alpha=1.0)
        img = t(image=image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Edge detection"


class ChangeColorspace(transform):
    """Transformacja zmieniająca przestrzeń kolorystyczną obrazu

    :param from_colorspace: Pierwotna przestrzeń kolorystyczna obrazu Allowed strings are: 
    "RGB", "BGR", "GRAY", "CIE", "YCrCb", "HSV", "HLS", "Lab", "Luv".
    :param to_colorspace: Przestrzeń kolorystyczna obrazu po zmianie
    """

    def __init__(self, from_colorspace: str, to_colorspace: str):
        if from_colorspace not in ["RGB", "BGR", "GRAY", "CIE", "YCrCb", "HSV", "HLS", "Lab", "Luv"]:
            ValueError("Allowed strings are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv")
        if to_colorspace not in ["RGB", "BGR", "GRAY", "CIE", "YCrCb", "HSV", "HLS", "Lab", "Luv"]:
            ValueError("Allowed strings are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv")
        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace

    def __call__(self, image, polygon, labels):
        t = iaa.ChangeColorspace(from_colorspace=self.from_colorspace,
                                 to_colorspace=self.to_colorspace)
        img = t(image=image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Change colorspace from " + self.from_colorspace + " to " + self.to_colorspace


class ExtractPolygon(transform):
    """Transformacja ekstachująca z obrazu sam obiekt (na podstawie podanych punktów wierzchołków obiektu)"""

    def __call__(self, image, polygon, labels):
        p = Polygon(polygon)
        # t = iaa.Pad(px=0)
        img = p.extract_from_image(image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Extract object"


class Crop(transform):
    """Transformacja przycinająca obraz"""

    def __call__(self, image, polygon, labels):

        h, w = image.shape[:2]
        top = max(0, min(row[1] for row in polygon))
        left = max(0, min(row[0] for row in polygon))
        bottom = max(0, h - max(row[1] for row in polygon))
        right = max(0, w - max(row[0] for row in polygon))

        t = iaa.Crop(px=(int(top), int(right), int(bottom), int(left)), keep_size=False)
        img = t(image=image)
        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] - left
            polygon[i][1] = polygon[i][1] - top

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Crop image"


class Normalize(transform):
    """Transformacja normalizująca obraz"""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, image, polygon, labels):
        t = transforms.Normalize(mean=self.mean, std=self.std)
        img = t(image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Normalize image mean=" + str(self.mean) + ", std=" + str(self.std)


class Rescale(transform):
    """Transformacja skalująca obraz"""

    def __init__(self, output_size: Tuple[int, int]):
        if not isinstance(output_size, (int, int)):
            ValueError("Output size must be (int, int)")
        if output_size[0] <= 0 or output_size[1] <= 0:
            ValueError("Output size must greater than 0")
        self.output_size = output_size

    def __call__(self, image, polygon, labels):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        t = iaa.Resize((new_h, new_w))
        img = t(image=image)

        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] * (new_w / w)
            polygon[i][1] = polygon[i][1] * (new_h / h)

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Rescale image to " + str(self.output_size)


class ToTensor(transform):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, polygon, labels):
        t = transforms.ToTensor()
        img = t(image)
        torch.FloatTensor(labels.values)
        torch.from_numpy(polygon)
        return {'image': img, 'polygon': torch.from_numpy(polygon), 'labels': torch.FloatTensor(labels.values)}

    def __str__(self):
        return "Convert to tensor"

