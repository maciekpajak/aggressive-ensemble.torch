from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np


def mAP_score(preds: pd.DataFrame, trues: pd.DataFrame):
    """ Funkcja obliczająca wynik modelu sieci neuronowej

        :param preds: Wartości przewidywane przez model
        :type preds: pd.DataFrame
        :param trues: Wartości prawdziwe
        :type trues: pd.DataFrame
        :return: Ogólny wynik modelu oraz wyniki dla każdej z cech w postaci listy
        :rtype: float, list
        """
    labels_score = []
    trues = trues.replace(-1,0)
    score = 0.0
    for p, t in zip(preds, trues):
        ap = 0.0
        if np.sum(trues[t]) != 0:
            ap = average_precision_score(trues[t], preds[p])
        labels_score.append(ap)
        score += ap

    score /= preds.shape[1]

    return score, labels_score
