import torch
import pandas as pd
import warnings
import os
import time
from src.aggressive_ensemble.Classifier import Classifier
from src.aggressive_ensemble.utils.general import rank_preds, merge_answers_by_rankings, merge_answers_by_probabilities


class Subensemble:
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
                 *classifiers):

        self.id = id

        if not labels:
            raise ValueError("Labels list cannot be empty")

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = device

        self.labels = labels
        self.classifiers = {c.id: c for c in classifiers}

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

        for classifier_id, classifier in self.classifiers.items():
            if not silent_mode:
                print("│ └╴" + classifier_id)

            save_dir = self.save_dir + classifier_id
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            (_, train_stats, val_stats) = classifier.train(data_dir=data_dir,
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
        print('Detecting...')
        print(self.id)
        answers = []
        for classifier_id, classifier in self.classifiers.items():

            if not silent_mode:
                print("│ └╴" + classifier_id)

            save_dir = self.save_dir + classifier_id + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            ans = classifier.detect(test_df=test_df,
                                    data_dir=data_dir,
                                    save_dir=save_dir,
                                    silent_mode=silent_mode)

            answers.extend(ans[self.labels])
            # end subensemble models detecting --------------------------------------


        answer_probabilities = merge_answers_by_probabilities(*answers)

        ranked_answers = [rank_preds(a).reset_index(drop=True) for a in answers]
        answer_ranking = merge_answers_by_rankings(*ranked_answers)


        return answer_probabilities, answer_ranking
