from Classifier import Classifier
from Ensemble import Ensemble
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from torch import nn
import Transforms.Transforms as tfms

ensemble_structure = {
    "subensemble1": {
        "labels": ["general_class_large vehicle", "general_class_small vehicle"],
        "models": ["model1"],
    },
    "subensemble2": {
        "labels": ["sunroof", "color_yellow"],
        "models": ["model1", "model2"],
    }
}

models_configs = {
    "model1": {
        "name": "resnet50_pretrained",
        "path": "H:/Studia/Praca inżynierska/basic_models/resnet50_pretrained.pth",
        "max_epochs": 1,
        "criterion": nn.BCELoss(),
        "batch_size": 4,
        "num_workers": 1,
        "preprocessing": [tfms.ExtractPolygon(),
                          tfms.RotateToHorizontal(),
                          tfms.ChangeColorspace("RGB", "HSV")],
        "augmentation": [tfms.RandomHorizontalFlip(), tfms.RandomVerticalFlip()],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "input_size": 224,
        "lr": 0.01,
        "momentum": 0.9,
        "val_every": 3,
        "feature_extract": False,
    },
    "model2": {
        "name": "resnet152_pretrained",
        "path": "H:/Studia/Praca inżynierska/basic_models/resnet152_pretrained.pth",
        "max_epochs": 1,
        "criterion": nn.BCELoss(),
        "batch_size": 4,
        "num_workers": 1,
        "preprocessing": [tfms.ExtractPolygon(),
                          tfms.RotateToHorizontal(),
                          tfms.ChangeColorspace("RGB", "HSV")],
        "augmentation": [tfms.RandomHorizontalFlip(), tfms.RandomVerticalFlip()],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "input_size": 224,
        "lr": 0.01,
        "momentum": 0.9,
        "val_every": 3,
        "feature_extract": False,
    }
}


def score(preds: pd.DataFrame, trues: pd.DataFrame):
    """ Funkcja obliczająca wynik modelu sieci neuronowej

        :param preds: Wartości przewidywane przez model
        :type preds: pd.DataFrame
        :param trues: Wartości prawdziwe
        :type trues: pd.DataFrame
        :return: Ogólny wynik modelu oraz wyniki dla każdej z cech w postaci listy
        :rtype: float, list
        """
    labels_score = []

    score = 0.0
    for p, t in zip(preds, trues):
        ap = 0.0
        if np.sum(trues[t]) != 0:
            ap = average_precision_score(trues[t], preds[p])
        labels_score.append(ap)
        score += ap

    score /= preds.shape[1]

    return score, labels_score


if __name__ == '__main__':
    device = "cpu"
    labels_csv = 'H:/Studia/Praca inżynierska/labels.csv'
    data_dir = 'H:/Studia/Praca inżynierska/data/train/'
    train_csv = 'H:/Studia/Praca inżynierska/train_short.csv'
    test_csv = 'H:/Studia/Praca inżynierska/test_short.csv'
    train_df = pd.DataFrame(pd.read_csv(train_csv))
    test_df = pd.DataFrame(pd.read_csv(test_csv))
    labels = list(pd.read_csv(labels_csv))
    for model in models_configs:
        print(models_configs[model])
    # models = {model: Classifier(labels, models_configs[model], device) for model in models_configs}
    ensemble = Ensemble(root_dir="H:/Studia/Praca inżynierska/",
                        labels=labels, models=models_configs, ensemble=ensemble_structure, max_subensemble_models=1,
                        mode="auto", device="cpu")
    print(ensemble)

    #ensemble.train(train_df=train_df, data_dir=data_dir, score_function=score)
    answer_probabilities, answer_01, answer_ranking  = ensemble.test(test_df=test_df, data_dir=data_dir)
    answer_ranking.to_csv(path_or_buf="H:/Studia/Praca inżynierska/answer.csv", index=False, header=True)

