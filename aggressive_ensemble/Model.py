import os
import types
import warnings

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt

import pandas as pd

import time
import pkbar

from sklearn.metrics import average_precision_score
import numpy as np

from Models.init_model import init_model
from Dataset.ImageDataset import ImageDataset
from Transforms.TransformImage import TransformImage


def create_dataset(input_size, mean, std, preprocessing, augmentations, csv_file, data_dir, labels):
    transform = TransformImage(input_size, mean, std, preprocessing, augmentations)
    return ImageDataset(csv_file, data_dir, labels, transform)


class Model:
    """ Klasa reprezentująca pojedynczy model sieci neuronowej.
    Oprócz samego modelu przechowuje wymagane atrybuty

    :param labels: lista możliwych etykiet
    :type labels: list
    :param model_config: konfiguracja modelu
    :type model_config: dict
    :param device: gpu or cpu
    :type device: string

    """

    def __init__(self, labels: list, model_config: dict, cpu_or_gpu: str):
        """Konstruktor klasy

        :param labels: lista możliwych etykiet
        :type labels: list
        :param model_config: konfiguracja modelu
        :type model_config: dict
        :param device: Rodziaj używanego procesora: gpu albo cpu
        :type device: string
        """

        if not labels:
            raise ValueError("Labels list cannot be empty")

        if not model_config:
            raise ValueError("Model_config dictionary cannot be empty")

        if cpu_or_gpu not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = cpu_or_gpu
        if cpu_or_gpu == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_config = model_config
        self.labels = labels

        self.model = torch.load(self.model_config['path'], map_location=torch.device(self.device))

        self.optimizer = optim.SGD(self.__get_params_to_update(), lr=model_config["lr"],
                                   momentum=model_config["momentum"])

        self.is_inception = (model_config["name"] == 'inception')

        self.params = {'batch_size': self.model_config["batch_size"],
                       'shuffle': True,
                       'num_workers': self.model_config["num_workers"]}

    def __str__(self):
        return str(self.model_config)

    def __get_params_to_update(self):
        """

        :return: Parameters to update
        :rtype: list
        """
        params_to_update = self.model.parameters()
        if self.model_config["feature_extract"]:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    # print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    pass
                    # print("\t", name)
        return params_to_update

    def save(self, path):
        """

        :param path: Ścieżka do zapisania modelu
        :type path: string
        :return:
        :rtype:
        """
        if not os.path.exists(path):
            raise ValueError("Path doesn't exist")

        torch.save(self.model, path)

    def train(self, train_csv: str, data_dir: str, score_function, criterion=nn.BCELoss()):
        """

        :return: Przetrenowany model
        :rtype:
        """

        if not os.path.exists(train_csv):
            raise ValueError("Train csv doesn't exist")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if not callable(criterion):
            raise ValueError("Criterion should be function")

        if not callable(score_function):
            raise ValueError("Score should be function")

        ratio = 0.8

        train_dataset = create_dataset(input_size=self.model_config['input_size'],
                                       mean=self.model_config['preprocessing']['normalization']['mean'],
                                       std=self.model_config['preprocessing']['normalization']['std'],
                                       preprocessing=self.model_config['preprocessing'],
                                       augmentations=self.model_config['augmentation'],
                                       csv_file=train_csv,
                                       data_dir=data_dir,
                                       labels=self.labels,
                                       )

        val_dataset = create_dataset(input_size=self.model_config['input_size'],
                                     mean=self.model_config['preprocessing']['normalization']['mean'],
                                     std=self.model_config['preprocessing']['normalization']['std'],
                                     preprocessing=self.model_config['preprocessing'],
                                     augmentations=None,
                                     csv_file=train_csv,
                                     data_dir=data_dir,
                                     labels=self.labels)

        l = int(len(train_dataset) * ratio)
        train_dataset = Subset(train_dataset, range(0, l))
        val_dataset = Subset(val_dataset, range(l, len(val_dataset)))

        dataloader = {"train": DataLoader(train_dataset, **self.params),
                      "val": DataLoader(val_dataset, **self.params)}

        since = time.time()

        max_epochs = self.model_config['max_epochs']
        val_every = self.model_config['val_every']

        headers = ['epoch', 'loss', 'score']
        headers.extend(self.labels)
        stats = {'train': pd.DataFrame(columns=headers),
                 'val': pd.DataFrame(columns=headers)}
        # stats_dir = stats_dir
        # os.mkdir(stats_dir)
        # stats_path = {'train': stats_dir + self.model_id + '_train_stats.csv',
        #              'val': stats_dir + self.model_id + '_val_stats.csv'}

        epoch_score_history = [0, 0, 0]
        for epoch in range(max_epochs):

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    bar = pkbar.Kbar(target=len(dataloader[phase]), epoch=epoch, num_epochs=max_epochs, width=50,
                                     always_stateful=True)
                else:
                    if (epoch + 1) % val_every != 0:
                        break
                    self.model.eval()
                    bar = pkbar.Kbar(target=len(dataloader[phase]), width=50,
                                     always_stateful=True)

                running_loss = 0.0

                preds = pd.DataFrame(columns=self.labels)
                trues = pd.DataFrame(columns=self.labels)
                for i, (tag_id, inputs, labels) in enumerate(dataloader[phase], 0):

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    preds = preds.append(pd.DataFrame(outputs.tolist(), columns=self.labels))
                    trues = trues.append(pd.DataFrame(labels.tolist(), columns=self.labels))

                    running_loss += loss.item()
                    score, _ = score_function(pd.DataFrame(outputs.tolist(), columns=self.labels),
                                     pd.DataFrame(labels.tolist(), columns=self.labels))
                    # bar update
                    bar.update(i, values=[("loss", loss.item()), ('score', score)])

                score, labels_score = score_function(preds, trues)
                epoch_loss = running_loss * inputs.size(0) / len(dataloader[phase].dataset)

                bar.add(1, values=[("epoch_loss", epoch_loss), ('epoch_score', score)])

                epoch_stats = [epoch + 1, epoch_loss, score]
                epoch_stats.extend(labels_score)
                stats[phase] = stats[phase].append(pd.DataFrame([epoch_stats], columns=headers), ignore_index=True)
                # stats[phase].to_csv(path_or_buf=stats_path[phase], index=False)

                if phase == 'train':
                    epoch_score_history[epoch % 3] = score

            if epoch_score_history[0] == epoch_score_history[1] and epoch_score_history[1] == epoch_score_history[2]:
                break
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model, stats

    def test(self, test_csv: str, data_dir: str) -> pd.DataFrame:
        """

        :return: Obiekt z odpowiedziami modelu na testowy zbiór danych
        :rtype: pd.DataFrame
        """
        if not os.path.exists(test_csv):
            raise ValueError("Test csv doesn't exist")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        test_dataset = create_dataset(input_size=self.model_config['input_size'],
                                      mean=self.model_config['preprocessing']['normalization']['mean'],
                                      std=self.model_config['preprocessing']['normalization']['std'],
                                      preprocessing=self.model_config['preprocessing'],
                                      augmentations=None,
                                      csv_file=test_csv,
                                      data_dir=data_dir,
                                      labels=self.labels)

        dataloader = DataLoader(test_dataset, **self.params)

        bar = pkbar.Kbar(target=len(dataloader), width=50, always_stateful=True)
        answer = pd.DataFrame(columns=self.labels)
        with torch.no_grad():
            self.model.eval()
            for i, (tag_id, inputs, labels) in enumerate(dataloader, 0):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                answer = answer.append(pd.DataFrame(outputs.tolist(), columns=self.labels, index=tag_id.tolist()))
                bar.update(i)
            bar.add(1)
        answer.index.name = 'tag_id'
        answer = self.rank_preds(answer)
        answer = answer.reset_index(drop=True)
        return answer

    @staticmethod
    def rank_preds(preds: pd.DataFrame) -> pd.DataFrame:
        """

        :param preds: Przewidywane wartości
        :type preds: pd.DataFrame
        :return: Uszeregowane przewiywane wratości od najbardziej prawdopodobnych do najmniej
        :rtype: pd.DataFrame
        """
        rpreds = pd.DataFrame(preds)
        for col in preds.columns.values:
            rpreds.loc[:, col] = preds.sort_values(by=col, ascending=False).index
        return rpreds

    def show_random_images(self, num: int) -> None:
        """

        :param num: Liczba obrazów do wyświetlenia
        :type num: int
        """

        indices = list(range(len(self.dataset['train'])))
        np.random.shuffle(indices)
        idx = indices[:num]
        from torch.utils.data.sampler import SubsetRandomSampler
        sampler = SubsetRandomSampler(idx)
        loader = DataLoader(self.dataset['train'], sampler=sampler, batch_size=num)
        dataiter = iter(loader)
        tag_id, images, labels = dataiter.next()
        to_pil = transforms.ToPILImage()
        fig = plt.figure(figsize=(20, 20))
        for ii in range(len(images)):
            image = to_pil(images[ii])
            fig.add_subplot(1, len(images), ii + 1)
            plt.axis('off')
            plt.imshow(image)
        plt.show()
