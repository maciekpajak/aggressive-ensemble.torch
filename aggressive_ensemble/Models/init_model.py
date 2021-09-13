from .models import *
from .models import __all__


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

    assert model_name in __all__, "Model doesn't exist. Possible models: {}".format(__all__)
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
    else:
        print("Model doesn't exist. Possible models: ")
        print(__all__)
        exit()

    return model, input_size, mean, std
