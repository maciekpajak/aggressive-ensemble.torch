from .models import *
from .models import __all__


def init_model(model_name, num_classes, feature_extract, use_pretrained=True):
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

    elif model_name == "inceptionV3":
        model, input_size, mean, std = inceptionV3(num_classes=num_classes,
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
