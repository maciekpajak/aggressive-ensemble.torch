import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional
from imgaug.augmentables.polys import Polygon
from imgaug import augmenters as iaa
from PIL import Image
from abc import ABCMeta, abstractmethod

__all__ = ['Resize',
           'RotateToHorizontal',
           'EdgeDetection',
           'Canny',
           'ChangeColorspace',
           'ExtractPolygon',
           'Crop',
           'Normalize',
           'ToTensor',
           'RandomVerticalFlip',
           'RandomHorizontalFlip',
           'RandomRotation',
           "RandomCrop"]


class ITransform(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, sample):
        return sample

    @abstractmethod
    def __repr__(self):
        return self.__class__.__name__


class RotateToHorizontal(ITransform):
    """Transformacja obracająca obiekt na obrazie do pozycji horyzontalnej"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        """

        :param sample:
        :type sample:
        :return:
        :rtype:
        """

        w, h = image.size
        y1, x1 = polygon[1, 1] - polygon[0, 1], polygon[1, 0] - polygon[0, 0]
        y2, x2 = polygon[2, 1] - polygon[1, 1], polygon[2, 0] - polygon[1, 0]
        object_w = max((np.sqrt(x1 * x1 + y1 * y1)), (np.sqrt(x2 * x2 + y2 * y2)))

        if (x1 * x1 + y1 * y1) > (x2 * x2 + y2 * y2):
            angle = np.arctan2(y1, x1) * 180 / np.pi
        else:
            angle = np.arctan2(y2, x2) * 180 / np.pi

        pad = max(0, int(10 - (w - object_w) / 2))
        image = functional.pad(img=image, padding=[pad], padding_mode='edge')
        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] + pad
            polygon[i][1] = polygon[i][1] + pad

        w, h = image.size
        image = functional.rotate(img=image, angle=angle)
        angle_rad = np.pi * angle / 180
        w_r, h_r = w / 2, -h / 2
        for i in range(polygon.shape[0]):
            x0, y0 = polygon[i][0], -polygon[i][1]
            polygon[i][0] = (x0 - w_r) * np.cos(angle_rad) - (y0 - h_r) * np.sin(angle_rad) + w_r
            polygon[i][1] = -((x0 - w_r) * np.sin(angle_rad) + (y0 - h_r) * np.cos(angle_rad) + h_r)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class EdgeDetection(ITransform):
    """Transformacja przeprowadzająca detekcję krawędzi"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        image = np.array(image)
        t = iaa.EdgeDetect(alpha=self.alpha)
        img = t(image=image)
        image = Image.fromarray(img)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class Canny(ITransform):
    """Transformacja przeprowadzająca detekcję krawędzi"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        image = np.array(image)
        t = iaa.Canny(alpha=self.alpha)
        img = t(image=image)
        image = Image.fromarray(img)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class ChangeColorspace(ITransform):
    """Transformacja zmieniająca przestrzeń kolorystyczną obrazu

    :param from_colorspace: Pierwotna przestrzeń kolorystyczna obrazu Allowed strings are:
    "RGB", "BGR", "GRAY", "CIE", "YCrCb", "HSV", "HLS", "Lab", "Luv".
    :param to_colorspace: Przestrzeń kolorystyczna obrazu po zmianie
    """

    def __init__(self, from_colorspace: str, to_colorspace: str):
        super().__init__()
        if from_colorspace not in ["RGB", "BGR", "GRAY", "CIE", "YCrCb", "HSV", "HLS", "Lab", "Luv"]:
            raise ValueError("Allowed strings are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv")
        if to_colorspace not in ["RGB", "BGR", "GRAY", "CIE", "YCrCb", "HSV", "HLS", "Lab", "Luv"]:
            raise ValueError("Allowed strings are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv")
        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = iaa.ChangeColorspace(from_colorspace=self.from_colorspace,
                                 to_colorspace=self.to_colorspace)
        image = np.array(image)
        img = t(image=image)
        image = Image.fromarray(img)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"({self.from_colorspace},{self.to_colorspace})"


class ExtractPolygon(ITransform):
    """Transformacja ekstachująca z obrazu sam obiekt (na podstawie podanych punktów wierzchołków obiektu)"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        p = Polygon(polygon)
        image = np.array(image)
        img = p.extract_from_image(image)
        image = Image.fromarray(img)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class Crop(ITransform):
    """Transformacja przycinająca obraz"""

    def __init__(self, top, left, height, width):
        super().__init__()
        if not isinstance(top, int) or not isinstance(left, int) or not isinstance(height, int) or not isinstance(width,
                                                                                                                  int):
            raise ValueError("Output size must be int")
        if top <= 0 or left <= 0 or height <= 0 or width <= 0:
            raise ValueError("Output size must greater than 0")
        self.left = left
        self.top = top
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]

        image = functional.crop(img=image, top=self.top, left=self.left, height=self.height, width=self.width)
        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] - self.left
            polygon[i][1] = polygon[i][1] - self.top
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"({self.top}, {self.left}, {self.height}, {self.width})"


class Normalize(ITransform):
    """Transformacja normalizująca obraz"""

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=self.mean, std=self.std),
                                transforms.ToPILImage()])
        image = t(image)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


class Resize(ITransform):
    """Transformacja skalująca obraz"""

    def __init__(self, h, w):
        super().__init__()
        if not isinstance(h, int) or not isinstance(w, int):
            raise ValueError("Output size must be int")
        if h <= 0 or w <= 0:
            raise ValueError("Output size must greater than 0")
        self.h = h
        self.w = w

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        w, h = image.size
        image = functional.resize(img=image, size=[self.h, self.w])

        for i in range(polygon.shape[0]):
            polygon[i][0] = polygon[i][0] * (self.w / w)
            polygon[i][1] = polygon[i][1] * (self.h / h)

        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(size=({self.h},{self.w}))"


class ToTensor(ITransform):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = transforms.ToTensor()
        image = t(image)
        sample = {'image': image, 'polygon': torch.from_numpy(polygon), 'labels': torch.FloatTensor(labels.values)}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlip(ITransform):
    """Transformacja losowo odbijająca obraz w pozycji horyzontalnej"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = transforms.RandomHorizontalFlip(p=0.5)
        image = t(image)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"


class RandomVerticalFlip(ITransform):
    """Transformacja losowo odbijająca obraz w pozycji wertykalnej """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = transforms.RandomVerticalFlip(p=0.5)
        image = t(image)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"


class RandomRotation(ITransform):
    """Transformacja obracająca obiekt o losowy kąt"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = transforms.RandomRotation(degrees=(0, 360))
        image = t(image)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class RandomCrop(ITransform):
    """Transformacja przycinająca obraz"""

    def __init__(self, h, w):
        super().__init__()
        if h <= 0 or w <= 0:
            ValueError("Output size must greater than 0")
        self.h = h
        self.w = w

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]
        t = transforms.RandomCrop(size=(self.h, self.w))
        image = t(image)
        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(size=({self.h},{self.w}))"
