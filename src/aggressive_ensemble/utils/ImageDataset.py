import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from typing import List


class ImageDataset(Dataset):
    """Klasa reprezentująca zbiór danych

    :param df:
    :type df:
    :param data_dir:
    :type data_dir:
    :param labels:
    :type labels:
    :param transform:
    :type transform:
    """

    def __init__(self, df: pd.DataFrame, data_dir: str, labels: List[str], transform=None):
        """
        :param data_dir:
        :type data_dir:
        :param labels:
        :type labels:
        :param transform:
        :type transform:
        """
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if labels is []:
            raise ValueError("Labels list cannot be empty")

        if transform and not callable(transform):
            raise ValueError("Transform should be a function")

        self.df = df
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        # self.df = self.df.replace(to_replace=-1, value=0)
        # self.df = self.df.reset_index()

    def __len__(self):
        """

        :return: Wielkość zbioru danych
        :rtype: int
        """
        return len(self.df)

    def __getitem__(self, idx):
        """

        :param idx: Indeks próbki
        :type idx:
        :return:
        :rtype:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tag_id = self.df['tag_id'][idx]
        img_id = self.df['image_id'][idx]

        polygon_points = []
        for i in range(1, 5, 1):
            px = 'p' + str(i) + '_x'
            py = 'p' + str(i) + '_y'
            if px in self.df.columns and py in self.df.columns:
                polygon_points.extend([px, py])
            else:
                break

        labels = self.df.loc[idx, self.labels]
        labels = labels.astype('int8')
        polygon = self.df.loc[idx, polygon_points]
        polygon = polygon.astype('float32')
        polygon = np.array([polygon])
        polygon = polygon.astype('int32').reshape(-1, 2)

        path = self.data_dir + str(img_id)
        if not os.path.exists(path):
            raise FileNotFoundError("There is no file {}".format(self.data_dir + str(img_id)))

        image = Image.open(path).convert('RGB')
        sample = {'image': image, 'polygon': polygon, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return tag_id, sample["image"], sample["labels"]
