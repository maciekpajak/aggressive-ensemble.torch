import copy

import torch

from src.aggressive_ensemble.Ensemble import Classifier, Ensemble
from src.aggressive_ensemble.utils.metrics import mAP_score
import src.aggressive_ensemble.utils.transforms as t
import src.aggressive_ensemble.utils.augmentations as a
from src.aggressive_ensemble.utils.general import show_random_images, check
import pandas as pd
from torch import nn

models_configs = {
    "resnet152_1": {
        "name": "resnet152_test3",
        "path": "H:/Studia/Praca inżynierska/basic_models/resnet152_pretrained.pth",
        "save_dir": "H:/Studia/Praca inżynierska/save/",
        "epochs": 6,
        "start_epoch": 0,
        "criterion": nn.BCELoss(),
        "batch_size": 32,
        "num_workers": 1,
        "preprocessing": [t.ExtractPolygon()],
        "augmentation": [a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "input_size": 224,
        "lr": 0.01,
        "momentum": 0.9,
        "val_every": 1,
        "autosave_every": 3,
        "shuffle": False,
        "feature_extract": True,
        "is_inception": False,
    },
    "resnet50_1": {
        "name": "resnet50_test3",
        "path": "H:/Studia/Praca inżynierska/basic_models/resnet50_pretrained.pth",
        "save_to": "H:/Studia/Praca inżynierska/basic_models/",
        "epochs": 6,
        "start_epoch": 0,
        "criterion": nn.BCELoss(),
        "batch_size": 32,
        "num_workers": 1,
        "preprocessing": [t.ExtractPolygon()],
        "augmentation": [a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "input_size": 224,
        "lr": 0.01,
        "momentum": 0.9,
        "val_every": 1,
        "autosave_every": 3,
        "shuffle": False,
        "feature_extract": True,
        "is_inception": False,
    }
}

if __name__ == '__main__':
    device = "cpu"
    root_dir = "H:/Studia/Praca inżynierska/"
    labels_csv = root_dir + 'labels.csv'
    data_dir = root_dir + 'data_cropped/data/'
    save_dir = root_dir + 'save/'
    train_csv = root_dir + 'data_cropped/train_cropped_he-short.csv'
    val_csv = root_dir + 'data_cropped/val_cropped_he-short.csv'
    test_csv = root_dir + 'data_cropped/test_cropped_he-short.csv'
    model_path = root_dir + 'basic_models/resnet152_pretrained.pth'
    train_df = pd.DataFrame(pd.read_csv(train_csv))
    val_df = pd.DataFrame(pd.read_csv(val_csv))
    test_df = pd.DataFrame(pd.read_csv(test_csv))
    labels = list(pd.read_csv(labels_csv))
    # print(labels)

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=37)
    resnet = torch.load('H:/Studia/Praca inżynierska/basic_models/resnet50_pretrained.pth')
    resnet50_1 = Classifier(name="resnet50_1",
                            labels=labels,
                            model=copy.deepcopy(resnet),
                            device="cpu",
                            feature_extract=True,
                            is_inception=False,
                            preprocessing=[t.ExtractPolygon()],
                            augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                            batch_size=5, num_workers=1, epochs=5, start_epoch=0,
                            input_size=224, lr=0.01, momentum=0.9, val_every=5,
                            save_every=5, shuffle=False, criterion=nn.BCELoss())
    resnet50_2 = Classifier(name="resnet50_2",
                            labels=labels,
                            model=copy.deepcopy(resnet),
                            device="cpu",
                            feature_extract=True,
                            is_inception=False,
                            preprocessing=[t.ExtractPolygon(), t.RotateToHorizontal()],
                            augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                            batch_size=5, num_workers=1, epochs=1, start_epoch=0,
                            input_size=224, lr=0.01, momentum=0.9, val_every=1,
                            save_every=1, shuffle=False, criterion=nn.BCELoss())

    # show_random_images(num=5,
    #                   df=train_df,
    #                   data_dir=data_dir,
    #                   preprocessing=[ExtractPolygon()],
    #                   augmentation=[RandomRotate()],
    #                   labels=labels)
    # model, train_stats, val_stats = classifier.train(train_df=train_df, val_df=val_df, score_function=mAP_score,
    #                                                 data_dir=data_dir,)

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
            "classifiers": [resnet50_1, resnet50_1, resnet50_2],
        },
        "subensemble2": {
            "labels": ['sunroof', 'luggage_carrier', 'open_cargo_area', 'enclosed_cab', 'spare_wheel',
                       'wrecked', 'flatbed', 'ladder', 'enclosed_box', 'soft_shell_box', 'harnessed_to_a_cart',
                       'ac_vents'],
            "classifiers": [resnet50_1, resnet50_1, resnet50_2],
        }
    }
    ensemble = Ensemble(id='Testensemble',
                        save_dir=save_dir,
                        labels=labels,
                        ensemble_structure=ensemble_structure,
                        device="cpu")

    ensemble.train(train_df=train_df, val_df=val_df, data_dir=data_dir, score_function=mAP_score)

    answer_probabilities, answer_01, answer_ranking = ensemble.detect(data_dir=data_dir, test_df=test_df,
                                                                      silent_mode=False)

    # a1, a2, a3 = pd.DataFrame(columns=labels), pd.DataFrame(columns=labels), pd.DataFrame(columns=labels)
    # for idx in test_df.index.values:
    #    print(f"%s/%s"%(idx,len(test_df.index.values)))
    #    answer_probabilities, answer_01, answer_ranking = ensemble.detect(data_dir=data_dir, test_df=test_df.loc[[idx]], silent_mode=True)
    #    a1 = a1.append(other=answer_probabilities, ignore_index=True)
    #    a2 = a2.append(other=answer_01, ignore_index=True)
    #    a3 = a3.append(other=answer_ranking, ignore_index=True)
    # ensemble = Ensemble(root_dir="H:/Studia/Praca inżynierska/",
    #                    labels=labels, classifiers=models_configs, ensemble=None, max_subensemble_models=2,
    #                    mode="manual", device="cpu")
    # print(ensemble)

    # ensemble.train(train_df=train_df, val_df=val_df, data_dir=data_dir, score_function=mAP_score)

    # ensemble.build_ensemble()
    # answer_probabilities, answer_01, answer_ranking = ensemble.test(test_df=test_df, data_dir=data_dir)
    # answer_ranking.to_csv(path_or_buf="H:/Studia/Praca inżynierska/answer.csv", index=False, header=True)
    # ans = pd.read_csv("H:/Studia/Praca inżynierska/save/Testensemble/answer_ranking.csv")
    # results = check(test_df=test_df, answer_df=ans, labels=labels)
