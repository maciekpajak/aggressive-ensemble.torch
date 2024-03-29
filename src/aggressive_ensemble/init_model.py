from .models import __all__
from .models import *


def init_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """

    :param model_name:
    :type model_name:
    :param num_classes:
    :type num_classes:
    :param feature_extract:
    :type feature_extract:
    :param use_pretrained:
    :type use_pretrained:
    :return: wygenerowny model, wielkość wejścia modelu, sugerowana średnia do normalizacji, sugerowane odchylenie standardowe do normalizacji
    :rtype: model,int,float,float
    """

    assert model_name in __all__, "Classifier doesn't exist. Possible classifiers: {}".format(__all__)
    model = None
    input_size = 0
    mean = [0, 0, 0]
    std = [1, 1, 1]

    if model_name == "resnet50":
        model, input_size, mean, std = resnet50(num_classes=num_classes,
                                                feature_extract=feature_extract,
                                                use_pretrained=use_pretrained)

    elif model_name == "resnet152":
        model, input_size, mean, std = resnet152(num_classes=num_classes,
                                                 feature_extract=feature_extract,
                                                 use_pretrained=use_pretrained)

    elif model_name == "alexnet":
        model, input_size, mean, std = alexnet(num_classes=num_classes,
                                               feature_extract=feature_extract,
                                               use_pretrained=use_pretrained)

    elif model_name == "vgg":
        model, input_size, mean, std = vgg(num_classes=num_classes,
                                           feature_extract=feature_extract,
                                           use_pretrained=use_pretrained)

    elif model_name == "densenet":
        model, input_size, mean, std = densenet(num_classes=num_classes,
                                                feature_extract=feature_extract,
                                                use_pretrained=use_pretrained)

    elif model_name == "inception":
        model, input_size, mean, std = inception(num_classes=num_classes,
                                                   feature_extract=feature_extract,
                                                   use_pretrained=use_pretrained)

    elif model_name == "xception":
        model, input_size, mean, std = xception(num_classes=num_classes,
                                                feature_extract=feature_extract,
                                                use_pretrained=use_pretrained)

    elif model_name == "nasnetamobile":
        model, input_size, mean, std = nasnetamobile(num_classes=num_classes,
                                                     feature_extract=feature_extract,
                                                     use_pretrained=use_pretrained)

    elif model_name == "nasnetalarge":
        model, input_size, mean, std = nasnetalarge(num_classes=num_classes,
                                                    feature_extract=feature_extract,
                                                    use_pretrained=use_pretrained)
    elif model_name == "yolov5s":
        model, input_size, mean, std = yolov5s(num_classes=num_classes,
                                                    feature_extract=feature_extract,
                                                    use_pretrained=use_pretrained)
    elif model_name == "yolov5m":
        model, input_size, mean, std = yolov5m(num_classes=num_classes,
                                                    feature_extract=feature_extract,
                                                    use_pretrained=use_pretrained)
    else:
        print("Classifier doesn't exist. Possible classifiers: ")
        print(__all__)
        exit()

    return model, input_size, mean, std
