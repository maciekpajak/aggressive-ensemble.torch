from typing import List

import numpy as np
import pandas as pd
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from .ImageDataset import ImageDataset
from .preprocessing import Rescale, ToTensor, Normalize


def rank_preds(preds: pd.DataFrame) -> pd.DataFrame:
    """

    :param preds: Przewidywane wartości
    :type preds: pd.DataFrame
    :return: Uszeregowane przewiywane wratości od najbardziej prawdopodobnych do najmniej
    :rtype: pd.DataFrame
    """
    if not isinstance(preds, pd.DataFrame):
        raise ValueError("Preds should be type pandas.DataFrame")

    if preds.empty:
        raise ValueError("Answers list cannot be empty")

    rpreds = preds.copy()
    for col in preds.columns.values:
        rpreds.loc[:, col] = preds.sort_values(by=col, ascending=False).index
    return rpreds


def merge_answers_by_rankings(*answers) -> pd.DataFrame:
    """

    :param answers:
    :type answers:
    :return:
    :rtype:
    """
    for a in answers:
        if not isinstance(a, pd.DataFrame):
            raise ValueError("Each answer in list should be type pandas.DataFrame")

    if not answers:
        raise ValueError("Answers list cannot be empty")

    tags = answers[0].iloc[:, 0].values
    columns = answers[0].columns.values

    dic = {col: pd.DataFrame(columns=tags, index=range(0, len(answers))) for col in columns}

    for ans, ans_idx in zip(answers, range(0, len(answers))):
        for col in ans:
            rank = 1
            for val in ans[col]:
                dic[col][val][ans_idx] = rank
                rank += 1

    answer = pd.DataFrame(columns=columns, index=tags)
    for d in dic:
        answer[d] = dic[d].mean()
    for col in columns:
        answer[col] = answer.sort_values(by=col, ascending=True).index
    answer = answer.reset_index(drop=True)

    return answer


def merge_answers_by_probabilities(*answers) -> pd.DataFrame:
    """

    :param answers:
    :type answers:
    :return:
    :rtype:
    """
    for a in answers:
        if not isinstance(a, pd.DataFrame):
            raise ValueError("Each answer in list should be type pandas.DataFrame")

    if not answers:
        raise ValueError("Answers list cannot be empty")

    mean = answers[0] - answers[0]
    for ans in answers:
        ans.sort_index(ascending=True)
        mean = mean + ans
    mean = mean / len(answers)

    return mean


def create_dataloader(data_dir,
                      dataframe,
                      labels,
                      preprocessing=None,
                      augmentation=None,
                      input_size=224,
                      mean=(0, 0, 0),
                      std=(1, 1, 1),
                      batch_size=32,
                      shuffle=True,
                      num_workers=1):
    if preprocessing is None:
        preprocessing = []
    if augmentation is None:
        augmentation = []

    adapt = [Rescale(output_size=(input_size, input_size)),
             ToTensor(),
             Normalize(mean=mean, std=std)]
    transform = transforms.Compose(preprocessing + augmentation + adapt)
    dataset = ImageDataset(dataframe, data_dir, labels, transform)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader


def show_random_images(num: int,
                       df: pd.DataFrame,
                       data_dir: str,
                       preprocessing,
                       augmentation,
                       labels,
                       input_size=224,
                       mean=(0, 0, 0),
                       std=(1, 1, 1), ) -> None:
    """

    :param num: Liczba obrazów do wyświetlenia
    :type num: int
    """
    loader = create_dataloader(data_dir,
                               dataframe=df,
                               labels=labels,
                               preprocessing=preprocessing,
                               augmentation=augmentation,
                               input_size=input_size,
                               mean=mean,
                               std=std,
                               batch_size=num,
                               shuffle=True,
                               num_workers=0)
    tag_id, images, labels = next(iter(loader))
    to_pil = transforms.ToPILImage()
    fig = plt.figure(figsize=(20, 20))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        fig.add_subplot(1, len(images), ii + 1)
        plt.axis('off')
        plt.imshow(image)
    plt.show()


def check(test_df: pd.DataFrame, answer_df: pd.DataFrame, labels: list):


    test_df= test_df.replace(to_replace=-1, value=0)
    test_df = test_df.set_index('tag_id')

    mAP = 0.0
    results = {}

    for label in labels:
        tp, p, K, ap = 0, 0, 0, 0
        for row in range(answer_df.shape[0]):

            tag = answer_df.loc[row, label]
            true = test_df.loc[tag, label]
            if true == 1:
                p += 1
                tp += 1
                K += 1
                ap += tp / p
            else:
                p += 1

        if K != 0:
            ap = ap / K
        else:
            ap = 0.0
        results[label] = ap

        mAP += ap

    mAP = mAP / len(labels)
    results["mAP"] = mAP

    return results
