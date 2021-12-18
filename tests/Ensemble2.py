import torch
import pandas as pd
import warnings
import os
import time
from src.aggressive_ensemble.Classifier import Classifier
from src.aggressive_ensemble.utils.general import rank_preds, merge_answers_by_rankings, merge_answers_by_probabilities


class Ensemble:
    """
    Klasa reprezentująca komitet sieci neuronowych


    """
    ensemble = {}
    models = {}

    def __init__(self,
                 id: str,
                 save_dir: str,
                 labels: list,
                 device: str = "cpu",
                 *subensembles):
        """Konstuktor klasy

        :param ensemble: Konfiguracja każdego modelu komitetu
        :type ensemble: dict
        """
        self.id = id

        if not labels:
            raise ValueError("Labels list cannot be empty")

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = device

        self.labels = labels
        self.ensemble = {s.id: s for s in subensembles}

        self.save_dir = save_dir + self.id + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(f"Outputs will be saved in {self.save_dir}")

    def __repr__(self):
        return self.ensemble

    def __str__(self):
        pass

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

        print("Training...")

        for subensemble_id, subensemble in self.ensemble.items():

            print("├╴" + subensemble_id)

            save_dir = self.save_dir + subensemble_id + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                subensemble.train(data_dir=data_dir,
                                  save_dir=save_dir,
                                  train_df=train_df,
                                  val_df=val_df,
                                  score_function=score_function)

    def detect(self,
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
        print('Detecting...')
        print(self.id)
        for subensemble_id, subensemble in self.ensemble.items():
            if not silent_mode:
                print("├╴" + subensemble_id)

            save_sub_dir = self.save_dir + subensemble_id + '/'
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

            answer_p, answer_r = subensemble.detect(test_df=test_df,
                                                    data_dir=data_dir,
                                                    silent_mode=silent_mode)

            answer_p.to_csv(save_sub_dir + "subensemble_answer_probabilities.csv")
            answer_r.to_csv(save_sub_dir + "subensemble_answer_ranking.csv", index=False)

            answer_probabilities = pd.concat([answer_probabilities, answer_p], axis=1)
            answer_ranking = pd.concat([answer_ranking, answer_r], axis=1)

            answer_01 = answer_probabilities > 0.5

            answer_probabilities.to_csv(self.save_dir + "ensemble_answer_probabilities.csv")
            answer_01.to_csv(self.save_dir + "ensemble_answer_01.csv")
            answer_ranking.to_csv(self.save_dir + "ensemble_answer_ranking.csv", index=False)

        return answer_probabilities, answer_01, answer_ranking
