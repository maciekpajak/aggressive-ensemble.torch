import json
import torch
import pandas as pd
import warnings

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

    def __init__(self, config, ensemble, models_config,  max_subensemble_models=1, mode="auto"):
        """Konstuktor klasy

        :param config: Konfiguracja komitetu
        :type config: dict
        :param ensemble: Konfiguracja każdego modelu komitetu
        :type ensemble: dict
        """
        # config = json.load(open(config_file))

        #self.config_example["device"] = config["device"]
        if mode not in ["manual","auto"]:
            raise ValueError("Mode should be either manual or auto")
        self.mode = mode

        if not isinstance(max_subensemble_models, int) or max_subensemble_models < 1:
            raise ValueError("Argument max_subensemble_models should be int and be greater than 1")
        self.max_subensemble_models = max_subensemble_models

        if config["device"] not in ["cpu","gpu"]:
            raise ValueError("Device should be either cpu or gpu")
        self.device = config["device"]
        if config["device"] == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.directories = config["directories"]
        self.mode = config["modes"]
        self.files = config["files"]

        self.labels = list(pd.read_csv(self.files["labels_csv"]))

        self.models= models
        self.ensemble = ensemble

        #self.models = self.__load_models()  # załadowanie modeli

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
        return self.ensemble

    def train(self):
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
                             model_config=self.models_config[model], device=self.device)
                for model in self.models_config}

        print("Training...")
        for model in self.models:
            self.models[model].show_random_images(5)
            self.models[model].train()
            self.models[model].save(self.directories["root_dir"] + "ensemble_models/" + model.model_id + ".pth")

    def test(self):
        """

        :return:
        :rtype:
        """
        print('Testing...')
        answers = {model: self.models[model].test() for model in self.models}
        answer = self.combine_answers(answers)
        return answer

    def autobuild_ensemble(self):
        """
        Automatycznie buduje komitet na podstawie statystyk wytrenowanych modeli
        Domyślnie przypisuje wszysktie modele do wszystkich cech
        :return:
        :rtype:
        """

        pass

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
                             model_config=self.models_config[model], device=self.device)
                for model in self.models_config}

    def __save_answer(self):
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
