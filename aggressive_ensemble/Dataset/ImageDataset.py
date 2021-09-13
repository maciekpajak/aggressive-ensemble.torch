import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path


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

    def __init__(self, csv_file, data_dir, labels, transform=None):
        """

        :param csv_file:
        :type csv_file:
        :param data_dir:
        :type data_dir:
        :param labels:
        :type labels:
        :param transform:
        :type transform:
        """
        self.df = pd.get_dummies(pd.read_csv(csv_file))
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.df = self.df.replace(to_replace=-1, value=0)

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
        for i in range(1, 100, 1):
            px = 'p' + str(i) + '_x'
            py = 'p' + str(i) + '_y'
            if px in self.df.columns and py in self.df.columns:
                polygon_points.extend([px, py])
            else:
                break

        labels = self.df.loc[idx, self.labels]
        polygon = self.df.loc[idx, polygon_points]
        polygon = np.array([polygon])
        polygon = polygon.astype('int').reshape(-1, 2)

        path = list(Path(self.data_dir).glob(str(img_id) + '.*'))[0]
        image = np.asarray(Image.open(path).convert('RGB'))
        sample = {'image': image, 'polygon': polygon, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return tag_id, sample['image'], sample["labels"]
