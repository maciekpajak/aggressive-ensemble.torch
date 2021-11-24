import torch
import pandas as pd
import warnings
import os

from .Classifier import Classifier
from .tools.rank_preds import rank_preds
from .tools.merge_answers import merge_answers_by_rankings, merge_answers_by_probabilities


class Ensemble:
    """
    Klasa reprezentująca komitet sieci neuronowych


    """

    def __init__(self, root_dir: str,
                 labels: list,
                 models: dict,
                 ensemble: dict = None,
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
            raise ValueError("tools cannot be empty")

        if mode == "manual":
            warnings.warn("Manual mode")

        if mode not in ["manual", "auto"]:
            raise ValueError("Mode should be either manual or auto")

        if not isinstance(max_subensemble_models, int) or max_subensemble_models < 1:
            raise ValueError("Argument max_subensemble_models should be int and be greater than 1")

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.mode = mode
        self.max_subensemble_models = max_subensemble_models
        self.device = device
        self.root_dir = root_dir
        self.labels = labels
        self.models = {m: Classifier(self.labels, models[m], self.device) for m in models}
        self.models_stats = {m: None for m in models}
        self.ensemble_stats = pd.DataFrame(columns=self.labels)
        self.ensemble = {} if ensemble is None else ensemble

        if not os.path.exists(root_dir + "ensemble_models/"):
            os.mkdir(root_dir + "ensemble_models/")

    def __str__(self):
        x = "Ensemble:\n"
        if self.ensemble is None:
            x = x + "empty"
        else:
            for sub in self.ensemble:
                x = x + "  " + sub + "\n\t\tlabels: \n"
                for l in self.ensemble[sub]["labels"]:
                    x = x + "\t\t\t" + l + "\n"
                x = x + "\t\tmodels: \n"
                for m in self.ensemble[sub]["models"]:
                    x = x + "\t\t\t" + m + "\n"
        return x

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, data_dir: str, score_function):
        """

        :return:
        :rtype:
        """
        if train_df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if not callable(score_function):
            raise ValueError("Score should be function")

        train_stats_path = self.root_dir + "training_stats/"
        if not os.path.exists(train_stats_path):
            os.mkdir(train_stats_path)
        print("Training stats will be saved in " + train_stats_path)

        print("Training...")
        for model in self.models:
            # zaladowanie modelu
            # m = Classifier(self.labels, self.models[model], self.device)

            # trening modelu
            print("Training model: " + model)
            (_, stats) = self.models[model].train(train_df, val_df, data_dir, score_function)
            self.models_stats[model] = stats
            # self.models_config[model.id]["path"] = self.root_dir + "ensemble_models/" + model.id + ".pth"  # zmiana sciezki modelu

            # zapisanie modelu
            if(self.models[model].save_to != ""):
                self.models[model].save(self.models[model].save_to + model + ".pth")
                print("Trained model saved: " + self.models[model].save_to + model + ".pth")
            else:
                torch.save(self.models[model].model, self.root_dir + "ensemble_models/" + model + ".pth")
                print("Trained model saved: " + self.root_dir + "ensemble_models/" + model + ".pth")

            # zapisanie statystyk modelu
            stats["train"].to_csv(path_or_buf=train_stats_path + model + "train_stats.csv", index=False,
                                  header=True)
            print("Trained model train_stats saved: " + train_stats_path + model + "train_stats.csv")

            stats["val"].to_csv(path_or_buf=train_stats_path + model + "val_stats.csv", index=False, header=True)
            print("Trained model val_stats saved: " + train_stats_path + model + "val_stats.csv")

    def test(self, test_df: pd.DataFrame, data_dir: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """

        :return:
        :rtype:
        """
        if test_df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        answer_probabilities = pd.DataFrame(columns=self.labels)
        answer_ranking = pd.DataFrame(columns=self.labels)

        answers_dir = {}
        print('Testing...')
        for subensemble in self.ensemble:
            for model in self.ensemble[subensemble]["models"]:

                print("Testing model: " + model)
                # zaladowanie modelu
                # m = Classifier(self.labels, self.models[model], self.device)
                if model in answers_dir.keys():
                    continue
                if model not in self.models.keys():
                    raise ValueError("There is no {} in models".format(model))
                ans = self.models[model].test(test_df, data_dir)
                answers_dir[model] = ans

        for subensemble in self.ensemble:
            answers = []
            print("Merging subensemble: " + subensemble + ":")
            for model in self.ensemble[subensemble]["models"]:
                print("\t" + model)
                ans = answers_dir[model]
                answers.extend([ans])

            answer1 = merge_answers_by_probabilities([a for a in answers])
            answer2 = merge_answers_by_rankings([rank_preds(a).reset_index(drop=True) for a in answers])
            subensemble_labels = self.ensemble[subensemble]["labels"]
            answer_probabilities[subensemble_labels] = answer1[subensemble_labels]
            answer_ranking[subensemble_labels] = answer2[subensemble_labels]

        print(answer_probabilities.head())
        answer_01 = answer_probabilities > 0.5
        print(answer_01.head())
        print(answer_ranking.head())

        return answer_probabilities, answer_01, answer_ranking

    def build_ensemble(self):
        """
        Automatycznie buduje komitet na podstawie statystyk wytrenowanych modeli
        Domyślnie przypisuje wszysktie modele do wszystkich cech
        :return:
        :rtype:
        """
        for stat in self.models_stats:
            last = self.models_stats[stat]["val"].xs(self.models_stats[stat]["val"].shape[0] - 1)
            last.name = stat
            self.ensemble_stats = self.ensemble_stats.append(last)
        i = 1
        for label in self.labels:
            tmp = self.ensemble_stats.sort_values(by=label, ascending=False)
            m = []
            for j in range(0, self.max_subensemble_models, 1):
                m = m + [tmp.index[j]]
            merged = False
            for sub in self.ensemble:
                if self.ensemble[sub]["models"] == m:
                    self.ensemble[sub]["labels"] = self.ensemble[sub]["labels"] + [label]
                    merged = True
            if not merged:
                self.ensemble["subensemble" + str(i)] = {"labels": [label], "models": m}
                i += 1

        print("Ensemble built")
        print(self)
