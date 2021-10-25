import types

import torch
import pandas as pd
import warnings
import os
from torch import nn
from typing import List

from aggressive_ensemble.Model import Model


class Ensemble:
    """
    Klasa reprezentująca komitet sieci neuronowych


    """

    def __init__(self, root_dir: str, labels: list, models: dict, ensemble: dict = None,
                 max_subensemble_models: int = 1,
                 mode: str = "auto",
                 device: str = "cpu"):
        """Konstuktor klasy

        :param ensemble: Konfiguracja każdego modelu komitetu
        :type ensemble: dict
        """
        if not os.path.exists(root_dir):
            raise ValueError("Root_dir doesn't exist")

        if not labels:
            raise ValueError("Labels list cannot be empty")

        if not models:
            raise ValueError("Models cannot be empty")

        if not models and mode == "manual":
            warnings.warn("Mode is manual and ensemble is not provided. Test function is not available in that case.")

        if mode not in ["manual", "auto"]:
            raise ValueError("Mode should be either manual or auto")
        self.mode = mode

        if not isinstance(max_subensemble_models, int) or max_subensemble_models < 1:
            raise ValueError("Argument max_subensemble_models should be int and be greater than 1")
        self.max_subensemble_models = max_subensemble_models

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")
        self.device = device

        self.root_dir = root_dir

        self.labels = labels

        self.models = models
        self.ensemble = ensemble

        if not os.path.exists(root_dir + "ensemble_models/"):
            os.mkdir(root_dir + "ensemble_models/")

    def __str__(self):
        return str(self.ensemble)

    def train(self, train_csv: str, data_dir: str, score_function, criterion=nn.BCELoss()):
        """

        :return:
        :rtype:
        """
        if not os.path.exists(train_csv):
            raise ValueError("Train csv doesn't exist")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if not callable(score_function):
            raise ValueError("Score should be function")

        if not callable(criterion):
            raise ValueError("Criterion should be function")

        train_stats_path = self.root_dir + "training_stats/"
        if not os.path.exists(train_stats_path):
            os.mkdir(train_stats_path)
        print("Training stats will be saved in " + train_stats_path)

        print("Training...")
        for model in self.models:
            # zaladowanie modelu
            m = Model(self.labels, self.models[model], self.device)

            # trening modelu
            print("Training model: " + model)
            (_, stats) = m.train(train_csv, data_dir, score_function, criterion)

            self.models[model]["path"] = self.root_dir + "ensemble_models/" + model + ".pth"  # zmiana sciezki modelu

            # zapisanie modelu
            torch.save(m.model, self.root_dir + "ensemble_models/" + model + ".pth")
            print("Trained model saved: " + self.root_dir + "ensemble_models/" + model + ".pth")

            # zapisanie statystyk modelu
            stats["train"].to_csv(path_or_buf=train_stats_path + model + "train_stats.csv", index=False,
                                  header=True)
            print("Trained model train_stats saved: " + train_stats_path + model + "train_stats.csv")

            stats["val"].to_csv(path_or_buf=train_stats_path + model + "val_stats.csv", index=False, header=True)
            print("Trained model val_stats saved: " + train_stats_path + model + "val_stats.csv")

    def test(self, test_csv: str, data_dir: str):
        """

        :return:
        :rtype:
        """
        if not os.path.exists(test_csv):
            raise ValueError("Test csv doesn't exist")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        final_answer = pd.DataFrame(columns=self.labels)
        print('Testing...')
        for subensemble in self.ensemble:
            answers = []
            print("Testing subensemble: " + subensemble)
            for model in self.ensemble[subensemble]["models"]:
                # zaladowanie modelu
                m = Model(self.labels, self.models[model], self.device)
                ans = m.test(test_csv, data_dir)
                answers.extend([ans])

            answer = self.combine_answers(answers)
            subensemble_labels = self.ensemble[subensemble]["labels"]
            final_answer[subensemble_labels] = answer[subensemble_labels]

        final_answer.to_csv(path_or_buf=self.root_dir + "answer.csv", index=False, header=True)
        print('Answer saved as ' + self.root_dir + "answer.csv")
        print(final_answer.head())

        return final_answer

    def autobuild_ensemble(self):
        """
        Automatycznie buduje komitet na podstawie statystyk wytrenowanych modeli
        Domyślnie przypisuje wszysktie modele do wszystkich cech
        :return:
        :rtype:
        """

        pass

    @staticmethod
    def combine_answers(answers: List[pd.DataFrame]) -> pd.DataFrame:
        """

        :param answers:
        :type answers:
        :return:
        :rtype:
        """
        # answers = list(answer.values())

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
