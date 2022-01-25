import torch
import pandas as pd
import warnings
import os
import time
from .Classifier import Classifier
from .utils.general import rank_preds, merge_answers_by_rankings, merge_answers_by_probabilities


class Ensemble:
    """
    Klasa reprezentująca komitet sieci neuronowych


    """

    def __init__(self,
                 id: str,
                 labels: list,
                 ensemble_structure: dict,
                 save_dir: str = os.getcwd(),
                 device: str = "cpu"):
        """Konstuktor klasy

        :param ensemble: Konfiguracja każdego modelu komitetu
        :type ensemble: dict
        """

        self.id = id

        if not isinstance(labels, list) or not isinstance(labels, tuple):
            raise ValueError("Labels should be list/tuple")
        if not labels:
            raise ValueError("Labels list cannot be empty")

        if isinstance(ensemble_structure, dict):
            raise ValueError("Ensemble structure should be dictionary")
        if ensemble_structure == {}:
            raise ValueError("Ensemble structure cannot be empty")

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = device

        self.labels = labels

        # self.classifiers = {}
        # for subensemble_name, subensemble_items in self.ensemble.items():
        #    for model in subensemble_items["classifiers"]:
        #        self.classifiers[model.id] = model

        # self.answers_dict = {c.id: None for c in classifiers}

        self.ensemble = ensemble_structure
        # self.ensemble_stats = pd.DataFrame(columns=self.labels)
        # self.ensemble = {} if ensemble is None else ensemble

        self.save_dir = save_dir + self.id + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(f"Outputs will be saved in {self.save_dir}")

    def __repr__(self):
        return self.ensemble

    def __str__(self):
        x = "Ensemble:\n"
        if self.ensemble is None:
            x = x + "empty"
        else:
            for sub in self.ensemble:
                x = x + "  " + sub + "\n\t\tlabels: \n"
                for l in self.ensemble[sub]["labels"]:
                    x = x + "\t\t\t" + l + "\n"
                x = x + "\t\tclassifiers: \n"
                for m in self.ensemble[sub]["classifiers"]:
                    x = x + "\t\t\t" + m + "\n"
        return x

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              data_dir: str,
              score_function,
              silent_mode=False):
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

        c_dict = {}
        print("Training...")

        for subensemble_name, subensemble_items in self.ensemble.items():
            if not silent_mode:
                print("├╴" + subensemble_name)

            save_sub_dir = self.save_dir + subensemble_name + '/'
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            # subensemble models detecting --------------------------------------------
            for model in subensemble_items["classifiers"]:

                if not silent_mode:
                    print("│ └╴" + model.id)

                save_dir = save_sub_dir + model.id + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                if model.id in c_dict.keys():
                    continue

                (_, train_stats, val_stats, _) = model.train(data_dir=data_dir,
                                                             save_dir=save_dir,
                                                             train_df=train_df,
                                                             val_df=val_df,
                                                             score_function=score_function)
                c_dict[model.id] = True
            # end subensemble models detecting --------------------------------------

    def __call__(self,
                 test_df: pd.DataFrame,
                 data_dir: str,
                 silent_mode=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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
        answer_01 = pd.DataFrame(columns=self.labels)
        answers_dict = {}
        if not silent_mode:
            print('Detecting...')
            print(self.id)
        for subensemble_name, subensemble_items in self.ensemble.items():
            if not silent_mode:
                print("├╴" + subensemble_name)
            answers = []
            subensemble_labels = subensemble_items["labels"]

            save_sub_dir = self.save_dir + subensemble_name + '/'
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            # subensemble models detecting --------------------------------------------
            for model in subensemble_items["classifiers"]:

                if not silent_mode:
                    print("│ └╴" + model.id)

                save_dir = save_sub_dir + model.id + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                if model.id in answers_dict.keys():
                    answers.extend([answers_dict[model.id][subensemble_labels]])
                    answers_dict[model.id].to_csv(save_dir + "answer.csv")
                    continue

                ans = model(test_df=test_df,
                            data_dir=data_dir,
                            save_dir=save_dir,
                            silent_mode=silent_mode)
                answers_dict[model.id] = ans
                answers.extend([ans[subensemble_labels]])
            # end subensemble models detecting --------------------------------------

            if not silent_mode:
                print("-- merging " + subensemble_name)

            answer1 = merge_answers_by_probabilities(*answers)

            ranked_answers = [rank_preds(a).reset_index(drop=True) for a in answers]
            answer2 = merge_answers_by_rankings(*ranked_answers)

            answer1.to_csv(save_sub_dir + "subensemble_answer_probabilities.csv")
            answer2.to_csv(save_sub_dir + "subensemble_answer_ranking.csv", index=False)

            answer_probabilities[subensemble_labels] = answer1[subensemble_labels]
            answer_ranking[subensemble_labels] = answer2[subensemble_labels]

            answer_probabilities.to_csv(self.save_dir + "ensemble_answer_probabilities.csv")
            answer_ranking.to_csv(self.save_dir + "ensemble_answer_ranking.csv", index=False)

        return answer_probabilities, answer_ranking
