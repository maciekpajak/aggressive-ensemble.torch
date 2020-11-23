from imgaug import augmenters as iaa


class RandomFlipUD(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = iaa.Flipud(p=0.5)
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}

class RandomFlipLR(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = iaa.Fliplr(p=0.5)
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}


class RandomRotate(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        t = iaa.Sequential([iaa.Pad(px=30),iaa.Affine(rotate=(0, 360))])
        img = t(image=image)

        return {'image': img, 'polygon': polygon, 'labels': labels}


class SwitchChannelsRGB(object):

    def __call__(self, sample):
        image, polygon, labels = sample['image'], sample['polygon'], sample['labels']

        #
        #
        #

        return {'image': image, 'polygon': polygon, 'labels': labels}

