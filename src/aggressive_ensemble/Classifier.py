import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models

from .ImageDataset import ImageDataset
from .transforms import Rescale, ToTensor, Normalize
from tqdm import tqdm
import shutil

NCOLS = shutil.get_terminal_size().columns


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


class Classifier:
    """ Klasa reprezentująca pojedynczy model sieci neuronowej.
    Oprócz samego modelu przechowuje wymagane atrybuty

    :param labels: lista możliwych etykiet
    :type labels: list
    :param model_config: konfiguracja modelu
    :type model_config: dict
    :param cpu_or_gpu: gpu or cpu
    :type cpu_or_gpu: string

    """

    def __init__(self,
                 name: str,
                 path: str,
                 labels: list,
                 device: str,
                 save_dir=None,
                 feature_extract=True,
                 is_inception=False):

        if not labels:
            raise ValueError("Labels list cannot be empty")

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = device
        if device == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir(save_dir):
            raise ValueError("Saving directory {} is not a directory or doesn't exist".format(save_dir))
        self.save_dir = save_dir

        self.id = name

        self.labels = labels

        self.feature_extract = feature_extract

        if not os.path.exists(path):
            raise ValueError("Provided model path {} doesn't exist".format(path))

        self.model = torch.load(path, map_location=torch.device(self.device))

        self.is_inception = is_inception

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.model

    def __get_params_to_update(self):
        """

        :return: Parameters to update
        :rtype: list
        """
        print("Parameters to update:")
        params_to_update = []
        # params_to_update = self.model.parameters()
        if self.feature_extract:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    pass
                    print("\t", name)
        return params_to_update

    def save(self, path):
        """

        :param path: Ścieżka do zapisania modelu
        :type path: string
        :return:
        :rtype:
        """
        torch.save(self.model, path)

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              score_function,
              data_dir: str,
              save_dir: str,
              labels,
              preprocessing,
              augmentation,
              mean,
              std,
              batch_size,
              num_workers,
              epochs=50,
              start_epoch=0,
              input_size=224,
              lr=0.01,
              momentum=0.9,
              val_every=1,
              save_every=3,
              shuffle=False,
              criterion=nn.BCELoss()
              ):
        """

        :return: Przetrenowany model
        :rtype:
        """

        if train_df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if not callable(score_function):
            raise ValueError("Score should be function")
        optimizer = optim.SGD(self.__get_params_to_update(), lr=lr,
                              momentum=momentum)

        train_loader = create_dataloader(data_dir=data_dir, dataframe=train_df, labels=labels,
                                         preprocessing=preprocessing, augmentation=augmentation,
                                         input_size=input_size, mean=mean, std=std, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

        val_loader = create_dataloader(data_dir=data_dir, dataframe=val_df, labels=labels,
                                       preprocessing=preprocessing, augmentation=None,
                                       input_size=input_size, mean=mean, std=std, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)
        since = time.time()

        headers = ['epoch', 'loss', 'score']
        headers.extend(self.labels)
        # stats dataframes
        train_stats = pd.DataFrame(columns=headers)
        val_stats = pd.DataFrame(columns=headers)

        # stats paths
        train_stats_path = self.save_dir + self.id + '_train_stats.csv'
        val_stats_path = self.save_dir + self.id + '_val_stats.csv'

        # best_fitness_from_train = 0.0
        # checkpoint = {'epoch': start_epoch,
        #              'model_state_dict': self.model.state_dict(),
        #              'optimizer_state_dict': optimizer.state_dict(),
        #              'loss': 0.0,
        #              'score': 0.0}
        best_fitness_from_val = 0.0
        best_val = {'epoch': start_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': 0.0,
                    'score': 0.0}

        for epoch in range(start_epoch, epochs):  # epoch -----------------------------------------

            self.model.train()
            running_loss = 0.0

            preds = pd.DataFrame(columns=self.labels)
            trues = pd.DataFrame(columns=self.labels)
            # progres bar
            nb = len(train_loader)
            print(('\n' + '%10s' * 3) % ('Epoch', 'loss', 'score'))
            pbar = tqdm(train_loader, total=nb, ncols=NCOLS,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            # batch ----------------------------------------------------------------------------
            for i, (tag_id, inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    if self.is_inception:
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = self.model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                preds = preds.append(pd.DataFrame(outputs.tolist(), columns=self.labels))
                trues = trues.append(pd.DataFrame(labels.tolist(), columns=self.labels))

                running_loss += loss.item()
                score, _ = score_function(pd.DataFrame(outputs.tolist(), columns=self.labels),
                                          pd.DataFrame(labels.tolist(), columns=self.labels))
                # bar update
                pbar.set_description(('%10s' * 1 + '%10.4g' * 2) % (f'{epoch + 1}/{epochs}', loss.item(), score))
            # end batch ---------------------------------------------------------------------------

            score, labels_score = score_function(preds, trues)
            epoch_loss = running_loss / len(train_loader)

            pbar.set_description(('%10s' * 1 + '%10.4g' * 2) % (f'{epoch + 1}/{epochs}', epoch_loss, score))

            # zapisywanie statystyk -------------------------------------------
            epoch_stats = [epoch + 1, epoch_loss, score]
            epoch_stats.extend(labels_score)
            train_stats = train_stats.append(pd.DataFrame([epoch_stats], columns=headers), ignore_index=True)
            train_stats.to_csv(path_or_buf=train_stats_path, index=False)

            final_epoch = (epoch + 1 == epochs)
            # walidacja -------------------------------------------
            noval = ((epoch + 1) % val_every != 0)
            if not noval or final_epoch:
                val_epoch_loss, val_score, val_labels_score = self.val(val_loader=val_loader,
                                                                       score_function=score_function,
                                                                       criterion=criterion)
                val_epoch_stats = [epoch + 1, val_epoch_loss, val_score]
                val_epoch_stats.extend(val_labels_score)
                val_stats = val_stats.append(pd.DataFrame([val_epoch_stats], columns=headers), ignore_index=True)
                val_stats.to_csv(path_or_buf=val_stats_path, index=False)
                if val_score > best_fitness_from_val:
                    best_val = {'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': val_epoch_loss,
                                'score': val_score}
                    best_fitness_from_val = val_score

            # autosave -------------------------------------------
            nosave = ((epoch + 1) % save_every != 0)
            if not nosave or final_epoch:  # if save
                print("Autosave...", end=" ")
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'score': score}, save_dir + self.id + "_autosave.pth")
                print("saved in " + save_dir + self.id + "_autosave.pth")

            # zapis najlepszego dopasowania---------------------------------
            # if score > best_fitness_from_train:
            #    checkpoint = {'epoch': epoch,
            #                  'model_state_dict': self.model.state_dict(),
            #                  'optimizer_state_dict': optimizer.state_dict(),
            #                  'loss': epoch_loss,
            #                  'score': score}
            #    best_fitness_from_train = score
            # if not nosave or final_epoch:
            #    print("Best fitness...", end=" ")
            #    torch.save(checkpoint, save_dir + self.id + "_bestfitness_from_training.pth")
            #   print("saved in " + save_dir + self.id + "_bestfitness_from_training.pth")

            if not nosave or final_epoch:
                print("Best fitness...", end=" ")
                torch.save(best_val, save_dir + self.id + "_bestfitness_from_validation.pth")
                print("saved in " + save_dir + self.id + "_bestfitness_from_validation.pth")
        # end epoch ------------------------------------------------------------------------------

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model, train_stats, val_stats

    def val(self, val_loader, score_function, criterion=nn.BCELoss()):

        if not callable(score_function):
            raise ValueError("Score should be function")

        self.model.eval()
        running_loss = 0.0

        preds = pd.DataFrame(columns=self.labels)
        trues = pd.DataFrame(columns=self.labels)

        nb = len(val_loader)
        pbar = tqdm(val_loader, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for i, (tag_id, inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

            preds = preds.append(pd.DataFrame(outputs.tolist(), columns=self.labels))
            trues = trues.append(pd.DataFrame(labels.tolist(), columns=self.labels))

            running_loss += loss.item()
            score, _ = score_function(pd.DataFrame(outputs.tolist(), columns=self.labels),
                                      pd.DataFrame(labels.tolist(), columns=self.labels))
            # bar update
            pbar.set_description(('%10s' + '%10.4g' * 2) % ("Val:", loss.item(), score))

        score, labels_score = score_function(preds, trues)
        epoch_loss = running_loss / len(val_loader)
        pbar.set_description(('%10s' + '%10.4g' * 2) % ("Val:", epoch_loss, score))

        return epoch_loss, score, labels_score

    def detect(self,
               test_df: pd.DataFrame,
               data_dir: str,
               save_dir: str,
               labels,
               preprocessing,
               mean,
               std,
               batch_size,
               num_workers,
               input_size=224,
               shuffle=False,
               ) -> pd.DataFrame:
        """

        :return: Obiekt z odpowiedziami modelu na testowy zbiór danych
        :rtype: pd.DataFrame
        """
        if test_df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        dataloader = create_dataloader(data_dir=data_dir, dataframe=test_df, labels=labels,
                                       preprocessing=preprocessing, augmentation=None,
                                       input_size=input_size, mean=mean, std=std, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)

        nb = len(dataloader)
        pbar = tqdm(dataloader, desc="Detecting", total=nb, ncols=NCOLS,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        answer = pd.DataFrame(columns=self.labels)
        with torch.no_grad():
            self.model.eval()
            for i, (tag_id, inputs, labels) in enumerate(pbar, 0):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                answer = answer.append(pd.DataFrame(outputs.tolist(), columns=self.labels, index=tag_id.tolist()))
        answer.index.name = 'tag_id'
        answer.to_csv(path_or_buf=save_dir + self.id + "_answer.csv")
        return answer

    def resume(self,
               PATH,
               train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               score_function,
               data_dir: str,
               save_dir: str,
               labels,
               preprocessing,
               augmentation,
               mean,
               std,
               batch_size,
               num_workers,
               epochs=50,
               input_size=224,
               lr=0.01,
               momentum=0.9,
               val_every=1,
               save_every=3,
               shuffle=False,
               criterion=nn.BCELoss()):

        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        if train_df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if not callable(score_function):
            raise ValueError("Score should be function")
        optimizer = optim.SGD(self.__get_params_to_update(), lr=lr,
                              momentum=momentum)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        train_loader = create_dataloader(data_dir=data_dir, dataframe=train_df, labels=labels,
                                         preprocessing=preprocessing, augmentation=augmentation,
                                         input_size=input_size, mean=mean, std=std, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

        val_loader = create_dataloader(data_dir=data_dir, dataframe=val_df, labels=labels,
                                       preprocessing=preprocessing, augmentation=None,
                                       input_size=input_size, mean=mean, std=std, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)
        since = time.time()

        headers = ['epoch', 'loss', 'score']
        headers.extend(self.labels)
        # stats dataframes
        train_stats = pd.DataFrame(columns=headers)
        val_stats = pd.DataFrame(columns=headers)

        # stats paths
        train_stats_path = self.save_dir + self.id + '_train_stats.csv'
        val_stats_path = self.save_dir + self.id + '_val_stats.csv'

        best_fitness = 0.0
        checkpoint = {'epoch': start_epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': 0.0,
                      'score': 0.0}

        for epoch in range(start_epoch, epochs):  # epoch -----------------------------------------

            self.model.train()

            running_loss = 0.0

            preds = pd.DataFrame(columns=self.labels)
            trues = pd.DataFrame(columns=self.labels)
            # progres bar
            nb = len(train_loader)
            print(('\n' + '%10s' * 3) % ('Epoch', 'loss', 'score'))
            pbar = tqdm(train_loader, total=nb, ncols=NCOLS,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            # batch ----------------------------------------------------------------------------
            for i, (tag_id, inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    if self.is_inception:
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = self.model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                preds = preds.append(pd.DataFrame(outputs.tolist(), columns=self.labels))
                trues = trues.append(pd.DataFrame(labels.tolist(), columns=self.labels))

                running_loss += loss.item()
                score, _ = score_function(pd.DataFrame(outputs.tolist(), columns=self.labels),
                                          pd.DataFrame(labels.tolist(), columns=self.labels))
                # bar update
                pbar.set_description(('%10s' * 1 + '%10.4g' * 2) % (f'{epoch + 1}/{epochs}', loss.item(), score))
            # end batch ---------------------------------------------------------------------------

            score, labels_score = score_function(preds, trues)
            epoch_loss = running_loss / len(train_loader)

            pbar.set_description(('%10s' * 1 + '%10.4g' * 2) % (f'{epoch + 1}/{epochs}', epoch_loss, score))

            # zapisywanie statystyk -------------------------------------------
            epoch_stats = [epoch + 1, epoch_loss, score]
            epoch_stats.extend(labels_score)
            train_stats = train_stats.append(pd.DataFrame([epoch_stats], columns=headers), ignore_index=True)
            train_stats.to_csv(path_or_buf=train_stats_path, index=False)

            final_epoch = (epoch + 1 == epochs)
            # walidacja -------------------------------------------
            noval = ((epoch + 1) % val_every != 0)
            if not noval or final_epoch:
                val_epoch_loss, val_score, val_labels_score = self.val(val_loader=val_loader,
                                                                       score_function=score_function,
                                                                       criterion=criterion)
                val_epoch_stats = [epoch + 1, val_epoch_loss, val_score]
                val_epoch_stats.extend(val_labels_score)
                val_stats = val_stats.append(pd.DataFrame([val_epoch_stats], columns=headers), ignore_index=True)
                val_stats.to_csv(path_or_buf=val_stats_path, index=False)

            # autosave -------------------------------------------
            nosave = ((epoch + 1) % save_every != 0)
            if not nosave or final_epoch:
                print("Autosave")
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'score': score}, save_dir + self.id + "_autosave.pth")

            # zapis najlepszego dopasowania---------------------------------
            if score > best_fitness:
                checkpoint = {'epoch': epoch,
                              'model_state_dict': self.model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'loss': loss,
                              'score': score}
                best_fitness = score
            if not nosave or final_epoch:
                torch.save(checkpoint, save_dir + self.id + "_bestfitness.pth")
        # end epoch ------------------------------------------------------------------------------

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model, train_stats, val_stats

    def show_random_images(self, num: int, train_df: pd.DataFrame, data_dir: str) -> None:
        """

        :param num: Liczba obrazów do wyświetlenia
        :type num: int
        """
        transform = transforms.Compose(self.preprocessing + self.augmentation + self.adapt)
        dataset = ImageDataset(train_df, data_dir, self.labels, transform)
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        idx = indices[:num]
        from torch.utils.data.sampler import SubsetRandomSampler
        sampler = SubsetRandomSampler(idx)
        loader = DataLoader(dataset, sampler=sampler, batch_size=num)
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
