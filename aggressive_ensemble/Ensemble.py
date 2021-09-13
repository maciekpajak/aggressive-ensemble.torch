import json
import torch
import pandas as pd

from aggressive_ensemble.Model import Model


def dict_pattern_match(pattern: dict, dict_to_check: dict):
    """

    :param pattern:
    :type pattern:
    :param dict_to_check:
    :type dict_to_check:
    :return:
    :rtype:
    """
    for k, v in pattern.items():
        assert k in dict_to_check.keys(), "Missing key '{k}' in provided config".format(k=k)
        if isinstance(v, dict):
            dict_pattern_match(v, dict_to_check[k])


class Ensemble:
    """
    Klasa reprezentująca komitet sieci neuronowych


    """

    config_example = {
        "directories": {
            "root_dir": "",
            "data_dir": "",
            "stats_dir": "",
            "models_dir": ""
        },
        "files": {
            "test_csv": "",
            "train_csv": "",
            "answer_csv": "",
            "labels_csv": ""
        },
        "modes": {
            "train": False,
            "test": False,
            "save_models": False
        },
        "device": "gpu"
    }
    ensemble_example = {
        "model1": {
            "name": "inception",
            "path": "",
            "max_epochs": 80,
            "criterion": "BCE",
            "batch_size": 32,
            "num_workers": 10,
            "preprocessing": {
                "polygon_extraction": True,
                "ratation_to_horizontal": True,
                "RGB_to_HSV": True,
                "edge_detection": False,
                "normalization": {
                    "mean": [0, 0, 0],
                    "std": [1, 1, 1]
                }
            },
            "augmentation": {
                "random_hflip": True,
                "random_vflip": True,
                "random_rotation": False,
                "switch_RGB_channel": False
            },
            "pretrained": True,
            "feature_extract": False,
            "input_size": 224,
            "lr": 0.01,
            "momentum": 0.9,
            "val_every": 3
        }
    }

    def __init__(self, config, ensemble):
        """Konstuktor klasy

        :param config: Konfiguracja komitetu
        :type config: dict
        :param ensemble: Konfiguracja każdego modelu komitetu
        :type ensemble: dict
        """
        # config = json.load(open(config_file))

        self.config_example["device"] = config["device"]
        self.device = config["device"]
        if config["device"] == "gpu":
            if not torch.cuda.is_available():
                print("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_example["directories"] = config["directories"]
        self.directories = config["directories"]
        self.config_example["modes"] = config["modes"]
        self.mode = config["modes"]
        self.config_example["files"] = config["files"]
        self.files = config["files"]

        self.labels = list(pd.read_csv(self.files["labels_csv"]))

        # self.ensemble = json.load(open(ensemble_config))
        self.ensemble = ensemble

        self.models = self.__load_models()  # załadowanie modeli

    def __call__(self):
        """

        :return:
        :rtype:
        """
        if self.mode["train"]:
            self.train()

        if self.mode["save_models"]:
            print('Saving')
            for model in self.models:
                self.models[model].save()

        if self.mode["test"]:
            answer = self.test()
            self.__save_answer(answer)

    def __str__(self):
        return ""

    def __load_models(self):
        """

        :return:
        :rtype:
        """
        print('Loading models in progress...')
        csv = {'test': self.files["test_csv"],
               'train': self.files["train_csv"]}
        dir = {'stats': self.directories["stats_dir"],
               'data': self.directories["data_dir"],
               'models': self.directories["models_dir"]}
        return {model: Model(csv=csv, dir=dir, labels=self.labels,
                             model_config=self.ensemble[model], device=self.device)
                for model in self.ensemble}

    def train(self):
        """

        :return:
        :rtype:
        """
        print("Training...")
        for model in self.models:
            self.models[model].show_random_images(5)
            self.models[model].train()
            self.models[model].save()

    def test(self):
        """

        :return:
        :rtype:
        """
        print('Testing...')
        answers = {model: self.models[model].test() for model in self.models}
        return self.combine_answers(answers)

    def __save_answer(self, answer):
        """

        :param answer:
        :type answer:
        :return:
        :rtype:
        """
        answer.to_csv(path_or_buf=self.files["answer_csv"], index=False, header=self.labels)
        print('Answer saved as ' + self.files["answer_csv"])

    def __save_model(self, model, model_path):
        """

        :param model:
        :type model:
        :param model_path:
        :type model_path:
        :return:
        :rtype:
        """
        self.models[model].save()

    @staticmethod
    def combine_answers(answer):
        """

        :param answer:
        :type answer:
        :return:
        :rtype:
        """


        print('Combining answers...')

        answers = list(answer.values())

        tags = answers[0].iloc[:, 0].values
        tmp = answers[0].copy(deep=False).set_index(tags)

        for col in tmp.columns.values:
            for tag in tags:
                rank = 0.0
                for ans in answers:
                    rank += ans[ans[col] == tag].index.values[0]
                tmp.loc[tag, col] = rank / len(answers)

        rpreds = tmp.copy(deep=False)
        for col in tmp.columns.values:
            rpreds.loc[:, col] = tmp.sort_values(by=col, ascending=True).index
        rpreds = rpreds.reset_index(drop=True)

        return rpreds
