import abc


class transform(metaclass=abs.ABCMeta):

    @abc.abstractmethod
    def __call__(self, image, polygon, labels):
        raise NotImplementedError

    def __str__(self):
        return "Not implemented transform"
