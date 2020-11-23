from torchvision import models
import torch
import pretrainedmodels

from torch import nn


class BasicModels:

    def __init__(self, model_name, num_classes, feature_extract, use_pretrained=True):

        self.model, self.input_size = self.__initialize_model(model_name, num_classes, feature_extract, use_pretrained)

    def __call__(self):
        return self.model, self.input_size

    def __set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def __initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 224

        elif model_name == "resnet152":
            """ Resnet152
            """
            model_ft = models.resnet152(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 299

        elif model_name == "xception":
            model_ft = pretrainedmodels.xception(num_classes=num_classes, pretrained='imagenet')
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.last_linear.in_features
            model_ft.last_linear = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 299

        elif model_name == "nasnetamobile":
            model_ft = pretrainedmodels.nasnetamobile(num_classes=num_classes, pretrained='imagenet')
            model_ft = torch.load("/content/drive/My Drive/PI/models/base_models/nasnetamobile-pretrained.pth")
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.last_linear.in_features
            model_ft.last_linear = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 331

        elif model_name == "nasnetalarge":
            model_ft = pretrainedmodels.nasnetalarge(num_classes=num_classes, pretrained='imagenet')
            model_ft = torch.load("/content/drive/My Drive/PI/models/base_models/nasnetamobile-pretrained.pth")
            self.__set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.last_linear.in_features
            model_ft.last_linear = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid())
            input_size = 224
        else:
            print("Invalid model name, exiting...")
            exit()
        return model_ft, input_size
