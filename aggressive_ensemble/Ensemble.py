import torch
import pandas as pd
import warnings
import os

from aggressive_ensemble.Model import Model


class Ensemble:
    """
    Klasa reprezentująca komitet sieci neuronowych


    """

    def __init__(self, root_dir: str, labels: list, models: dict, ensemble=None, max_subensemble_models=1, mode="auto",
                 device="cpu"):
        """Konstuktor klasy

        :param ensemble: Konfiguracja każdego modelu komitetu
        :type ensemble: dict
        """
        if not os.path.exists(root_dir):
            raise ValueError("Root_dir doesn't exist")

        if len(labels) == 0:
            raise ValueError("No labels")

        if mode not in ["manual", "auto"]:
            raise ValueError("Mode should be either manual or auto")
        self.mode = mode

        if not isinstance(max_subensemble_models, int) or max_subensemble_models < 1:
            raise ValueError("Argument max_subensemble_models should be int and be greater than 1")
        self.max_subensemble_models = max_subensemble_models

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")
        self.device = device
        if self.device == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.root_dir = root_dir

        self.labels = labels

        self.models = models
        self.ensemble = ensemble

        if not os.path.exists(root_dir + "ensemble_models/"):
            os.mkdir(root_dir + "ensemble_models/")

    def __str__(self):
        return str(self.ensemble)

    def train(self, train_csv, data_dir):
        """

        :return:
        :rtype:
        """
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
            (_, stats) = m.train(train_csv, data_dir)

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

    def test(self, test_csv, data_dir):
        """

        :return:
        :rtype:
        """
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
    def combine_answers(answers):
        """

        :param answers:
        :type answers:
        :return:
        :rtype:
        """

        print('Combining answers...')

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
