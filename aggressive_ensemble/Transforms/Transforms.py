import torch
import numpy as np
import random

from torchvision import transforms
from imgaug.augmentables.polys import Polygon
from imgaug import augmenters as iaa

__all__ = ['Rescale', 'RotateToHorizontal', 'EdgeDetection', 'ChangeColorspace', 'ExtractPolygon', 'Crop',
           'Normalize', 'ToTensor', 'RandomVerticalFlip', 'RandomHorizontalFlip', 'RandomRotate', 'SwitchRGBChannels']


class RotateToHorizontal(object):
    """Transformacja obracająca obiekt na obrazie do pozycji horyzontalnej"""
    def __call__(self, sample):
        """

        :param sample:
        :type sample:
        :return:
        :rtype:
        """
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

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


class EdgeDetection(object):
    """Transformacja przeprowadzająca detekcję krawędzi"""
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        t = iaa.EdgeDetect(alpha=1.0)
        img = t(image=image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Edge detection"


class ChangeColorspace(object):
    """Transformacja zmieniająca przestrzeń kolorystyczną obrazu

    :param from_colorspace: Pierwotna przestrzeń kolorystyczna obrazu
    :param to_colorspace: Przestrzeń kolorystyczna obrazu po zmianie
    """
    def __init__(self, from_colorspace, to_colorspace):
        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        t = iaa.ChangeColorspace(from_colorspace=self.from_colorspace,
                                 to_colorspace=self.to_colorspace)
        img = t(image=image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Change colorspace from " + self.from_colorspace + " to " + self.to_colorspace


class ExtractPolygon(object):
    """Transformacja ekstachująca z obrazu sam obiekt (na podstawie podanych punktów wierzchołków obiektu)"""
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        p = Polygon(polygon)
        t = iaa.Pad(px=5)
        img = t(image=p.extract_from_image(image))
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Extract object"


class Crop(object):
    """Transformacja przycinająca obraz"""
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        h, w = image.shape[:2]
        top = max(0, min(row[1] for row in polygon))
        left = max(0, min(row[0] for row in polygon))
        bottom = max(0, h - max(row[1] for row in polygon))
        right = max(0, w - max(row[0] for row in polygon))

        t = iaa.Crop(px=(int(top), int(right), int(bottom), int(left)))
        img = t(image=image)
        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] - left
            polygon[i][1] = polygon[i][1] - top

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Crop image"


class Normalize(object):
    """Transformacja normalizująca obraz"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        t = transforms.Normalize(mean=self.mean, std=self.std)
        img = t(image)
        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Normalize image mean=" + self.mean + ", std=" + self.std


class Rescale(object):
    """Transformacja skalująca obraz"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

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
        return "Rescale image to " + self.output_size


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = transforms.ToTensor()
        img = t(image)
        return {'image': img, 'polygon': torch.from_numpy(polygon), 'labels': torch.FloatTensor(labels.values)}

    def __str__(self):
        return "Convert to tensor"


class RandomHorizontalFlip(object):
    """Transformacja losowo odbijająca obraz w pozycji horyzontalnej"""
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = iaa.Fliplr(p=0.5)
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Random horizonatal flip"


class RandomVerticalFlip(object):
    """Transformacja losowo odbijająca obraz w pozycji wertykalnej """
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = iaa.Flipud(p=0.5)
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Random vertical flip"


class RandomRotate(object):
    """Transformacja obracająca obiekt o losowy kąt"""
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = iaa.Sequential([iaa.Affine(rotate=(0, 360))])
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Random rotation"


class SwitchRGBChannels(object):
    """Transformacja zamieniająca kanały RGB"""
    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        x = bool(random.getrandbits(1))
        if labels["color_red"] == 1:
            labels["color_red"] = 0.0
            if x:
                # switch red channel with blue channel
                image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
                labels["color_blue"] = 1.0
            else:
                # switch red channel with green channel
                image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
                labels["color_green"] = 1.0
        elif labels["color_green"] == 1:
            labels["color_green"] = 0.0
            if x:
                # switch green channel with blue channel
                image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
                labels["color_blue"] = 1.0
            else:
                # switch green channel with red channel
                image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
                labels["color_green"] = 1.0
        elif labels["color_blue"] == 1:
            labels["color_blue"] = 0.0
            if x:
                # switch red channel with blue channel
                image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
                labels["color_red"] = 1.0
            else:
                # switch blue channel with green channel
                image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
                labels["color_green"] = 1.0

        return {'image': image, 'polygon': polygon, 'labels': labels}

    def __str__(self):
        return "Switch RGB channels"
