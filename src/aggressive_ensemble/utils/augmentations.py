
import torch
import numpy as np
import random

from imgaug import augmenters as iaa

__all__ = ['RandomVerticalFlip', 'RandomHorizontalFlip', 'RandomRotate', 'SwitchRGBChannels']



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


class RandomCrop(object):
    """Transformacja przycinająca obraz"""

    def __init__(self, size_x, size_y):

        if size_x[0] <= 0 or size_y[1] <= 0:
            ValueError("Output size must greater than 0")


    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

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