import random
import warnings

import numpy as np
from imgaug import augmenters as iaa
from PIL import Image


class CropFromImage:
    """Transformacja przycinająca obraz"""

    def __init__(self, size_x, size_y):

        if size_x[0] <= 0 or size_y[1] <= 0:
            ValueError("Output size must greater than 0")

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]

        w, h = image.size
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


class SwitchRGBChannels:
    """Transformacja zamieniająca kanały RGB"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]

        x = False
        if random.randrange(0, 100) <= 100 * self.p:
            x = True
        y = bool(random.getrandbits(1))

        if x:
            image = np.array(image)
            if labels["color_red"] == 1:
                labels["color_red"] = 0.0
                if y:
                    # switch red channel with blue channel
                    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
                    labels["color_blue"] = 1.0
                else:
                    # switch red channel with green channel
                    image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
                    labels["color_green"] = 1.0
            elif labels["color_green"] == 1:
                labels["color_green"] = 0.0
                if y:
                    # switch green channel with blue channel
                    image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
                    labels["color_blue"] = 1.0
                else:
                    # switch green channel with red channel
                    image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
                    labels["color_green"] = 1.0
            elif labels["color_blue"] == 1:
                labels["color_blue"] = 0.0
                if y:
                    # switch red channel with blue channel
                    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
                    labels["color_red"] = 1.0
                else:
                    # switch blue channel with green channel
                    image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
                    labels["color_green"] = 1.0
            image = Image.fromarray(image)

        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"
