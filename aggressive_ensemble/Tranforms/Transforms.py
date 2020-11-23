import torch
import numpy as np

from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon


class RotateToHorizontal(object):

    def __call__(self, sample):

        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        h, w = image.shape[:2]
        y1, x1 = polygon[1, 1] - polygon[0, 1], polygon[1, 0] - polygon[0, 0]
        y2, x2 = polygon[2, 1] - polygon[1, 1], polygon[2, 0] - polygon[1, 0]
        object_w = max((np.sqrt(x1 * x1 + y1 * y1)), (np.sqrt(x2 * x2 + y2 * y2)))

        if (x1 * x1 + y1 * y1) > (x2 * x2 + y2 * y2):
            angle = np.arctan2(y1, x1) * 180 / np.pi
        else:
            angle = np.arctan2(y2, x2) * 180 / np.pi

        pad = max(0, int(20 - (w - object_w) / 2))
        t = iaa.Sequential([iaa.Pad(px=pad), iaa.Affine(rotate=-angle)])
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}


class EdgeDetection(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        t = iaa.EdgeDetect(alpha=1.0)
        img = t(image=image)
        return {'image': img, 'polygon': polygon, 'labels': labels}


class ChangeColorspace(object):

    def __init__(self, from_colorspace, to_colorspace):
        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        t = iaa.ChangeColorspace(from_colorspace=self.from_colorspace,
                                 to_colorspace=self.to_colorspace)
        img = t(image=image)
        return {'image': img, 'polygon': polygon, 'labels': labels}


class ExtractPolygon(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']
        p = Polygon(polygon)
        img = p.extract_from_image(image)
        return {'image': img, 'polygon': polygon, 'labels': labels}


class Crop(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        h, w = image.shape[:2]
        top = max(0, min(row[1] for row in polygon))
        left = max(0, min(row[0] for row in polygon))
        bottom = max(0, h - max(row[1] for row in polygon))
        right = max(0, w - max(row[0] for row in polygon))

        t = iaa.Crop(px=(top, right, bottom, left))
        img = t(image=image)
        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] - left
            polygon[i][1] = polygon[i][1] - top

        return {'image': img, 'polygon': polygon, 'labels': labels}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = transforms.Normalize(mean=self.mean, std=self.std)
        img = t(image)
        return {'image': img, 'polygon': polygon, 'labels': labels}


class Rescale(object):

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

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        polygon = polygon * [new_w / w, new_h / h]

        return {'image': img, 'polygon': polygon, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = transforms.ToTensor()
        img = t(image)
        return {'image': img, 'polygon': torch.from_numpy(polygon), 'labels': torch.FloatTensor(labels.values)}
