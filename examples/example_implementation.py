from src.aggressive_ensemble.Ensemble import Ensemble, Classifier
from src.aggressive_ensemble.tools.mAP_score import mAP_score
from src.aggressive_ensemble.transforms import *
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from torch import nn
import torchvision.models as models

ensemble_structure = {
    "subensemble1": {
        "labels": ['general_class_large vehicle', 'general_class_small vehicle', 'sub_class_bus',
                   'sub_class_cement mixer', 'sub_class_crane truck', 'sub_class_dedicated agricultural vehicle',
                   'sub_class_hatchback', 'sub_class_jeep', 'sub_class_light truck', 'sub_class_minibus',
                   'sub_class_minivan',
                   'sub_class_pickup', 'sub_class_prime mover', 'sub_class_sedan', 'sub_class_tanker',
                   'sub_class_truck',
                   'sub_class_van', 'color_black', 'color_blue', 'color_green', 'color_other', 'color_red',
                   'color_silver/grey', 'color_white', 'color_yellow'],
        "models": ["resnet50", "resnet152_1"],
    },
    "subensemble2": {
        "labels": ['sunroof', 'luggage_carrier', 'open_cargo_area', 'enclosed_cab', 'spare_wheel',
                   'wrecked', 'flatbed', 'ladder', 'enclosed_box', 'soft_shell_box', 'harnessed_to_a_cart',
                   'ac_vents'],
        "models": ["resnet152_1"],
    }
}

models_configs = {
    "resnet152_1": {
        "name": "resnet152_test3",
        "path": "H:/Studia/Praca inżynierska/basic_models/resnet152_pretrained.pth",
        "save_to": "H:/Studia/Praca inżynierska/basic_models/",
        "max_epochs": 6,
        "criterion": nn.BCELoss(),
        "batch_size": 32,
        "num_workers": 1,
        "preprocessing": [ExtractPolygon()],
        "augmentation": [RandomHorizontalFlip(), RandomVerticalFlip()],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "input_size": 224,
        "lr": 0.01,
        "momentum": 0.9,
        "val_every": 1,
        "autosave_every": 3,
        "feature_extract": True,
    }
}

if __name__ == '__main__':
    device = "cpu"
    labels_csv = 'H:/Studia/Praca inżynierska/labels.csv'
    data_dir = 'H:/Studia/Praca inżynierska/data_cropped/data/'
    save_dir = 'H:/Studia/Praca inżynierska/save/'
    train_csv = 'H:/Studia/Praca inżynierska/data_cropped/train_cropped_he-short.csv'
    val_csv = 'H:/Studia/Praca inżynierska/data_cropped/val_cropped_he-short.csv'
    test_csv = 'H:/Studia/Praca inżynierska/data_cropped/test_cropped_he-short.csv'
    train_df = pd.DataFrame(pd.read_csv(train_csv))
    val_df = pd.DataFrame(pd.read_csv(val_csv))
    test_df = pd.DataFrame(pd.read_csv(test_csv))
    labels = list(pd.read_csv(labels_csv))
    # print(labels)

    c = Classifier(name="resnet50_testtest",
                   labels=labels,
                   path="H:/Studia/Praca inżynierska/basic_models/resnet152_pretrained.pth",
                   device="cpu",
                   save_dir="H:/Studia/Praca inżynierska/save/",
                   feature_extract=True,
                   is_inception=False)

    print(c)

    p = [ExtractPolygon()]
    a = [RandomHorizontalFlip(), RandomVerticalFlip()]
    c.train(train_df=train_df, val_df=val_df, score_function=mAP_score, data_dir=data_dir, save_dir=save_dir,
            labels=labels, preprocessing=p, augmentation=a, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            batch_size=5, num_workers=1, epochs=50, start_epoch=0, input_size=224, lr=0.01, momentum=0.9, val_every=5,
            save_every=5, shuffle=True, criterion=nn.BCELoss())
    # ensemble = Ensemble(root_dir="H:/Studia/Praca inżynierska/",
    #                    labels=labels, models=models_configs, ensemble=None, max_subensemble_models=2,
    #                    mode="manual", device="cpu")
    # print(ensemble)

    # ensemble.train(train_df=train_df, val_df=val_df, data_dir=data_dir, score_function=mAP_score)

    # ensemble.build_ensemble()
    # answer_probabilities, answer_01, answer_ranking = ensemble.test(test_df=test_df, data_dir=data_dir)
    # answer_ranking.to_csv(path_or_buf="H:/Studia/Praca inżynierska/answer.csv", index=False, header=True)
