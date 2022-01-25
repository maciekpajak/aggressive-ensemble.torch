import os
import time
import warnings

import pandas as pd
import torch
from torch import optim, nn

from tqdm import tqdm
import shutil

from .utils.general import create_dataloader

NCOLS = shutil.get_terminal_size().columns


class Classifier:
    """ Klasa reprezentująca pojedynczy model sieci neuronowej.
    Oprócz samego modelu przechowuje wymagane atrybuty

    """

    def __init__(self,
                 name: str,             # nazwa klasyfikatora
                 model,                 # model bazowy
                 labels: list,          # lista cech
                 device: str = "cpu",   # gpu lub cpu
                 preprocessing=None,      # transformacje preprocessingu
                 augmentation=None,       # transformacje augmentacji
                 mean=(0, 0, 0),        # średnia do normalizacji
                 std=(1, 1, 1),         # odchylenie standardowe do normalizacji
                 batch_size=32,         # wielkość paczki danych
                 num_workers=1,         # liczba wątków
                 epochs=50,             # maksymalna liczba epok treningu
                 start_epoch=0,         # numer startowej epoki
                 input_size=224,        # wielkość wejścia
                 lr=0.01,               # współczynnik lr
                 momentum=0.9,          # współczynnik momentum
                 val_every=1,           # co ile epok walidacja
                 save_every=3,          # co ile epok zapisywanie aktualnego stanu modelu
                 shuffle=False,         # czy przetasować dane wejściowe
                 criterion=nn.BCELoss(),# funkcja strat
                 feature_extract=True,
                 is_inception=False,    # czy model jest Inception V3
                 checkpoint_path=None   # ścieżka do trenowanego już modelu
                 ):

        if augmentation is None:
            augmentation = []
        if preprocessing is None:
            preprocessing = []

        self.id = name

        if not isinstance(labels, list) or not isinstance(labels, tuple):
            raise ValueError("Labels should be list/tuple")
        if not labels:
            raise ValueError("Labels list cannot be empty")

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = device
        if device == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labels = labels

        self.feature_extract = feature_extract

        self.model = model.to(self.device)

        self.is_inception = is_inception

        self.labels = labels
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.input_size = input_size
        self.lr = lr
        self.momentum = momentum
        self.val_every = val_every
        self.save_every = save_every
        self.shuffle = shuffle
        self.criterion = criterion
        self.optimizer = optim.SGD(self.__get_params_to_update(), lr=self.lr,
                                   momentum=self.momentum)

        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise ValueError("Checkpoint path doesn't exist")
            else:
                self.load_checkpoint(checkpoint_path)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.model

    def load_checkpoint(self, path):

        if not os.path.exists(path):
            raise ValueError("Checkpoint pathdoesn't exist")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']

    def __get_params_to_update(self):
        """

        :return: Parameters to update
        :rtype: list
        """

        params_to_update = []
        if self.feature_extract:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    pass

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
              data_dir: str,
              save_dir: str,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              score_function,
              patience=2
              ):
        """

        :return: Przetrenowany model
        :rtype:
        """

        if train_df.empty:
            raise ValueError("DataFrame train_df cannot be empty")

        if val_df.empty:
            raise ValueError("DataFrame val_df cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        if not os.path.exists(save_dir):
            raise ValueError("Save dir doesn't exist")

        if not callable(score_function):
            raise ValueError("Score should be a function")

        train_loader = create_dataloader(data_dir=data_dir, dataframe=train_df, labels=self.labels,
                                         preprocessing=self.preprocessing, augmentation=self.augmentation,
                                         input_size=self.input_size, mean=self.mean, std=self.std,
                                         batch_size=self.batch_size,
                                         shuffle=self.shuffle, num_workers=self.num_workers)

        val_loader = create_dataloader(data_dir=data_dir, dataframe=val_df, labels=self.labels,
                                       preprocessing=self.preprocessing, augmentation=None,
                                       input_size=self.input_size, mean=self.mean, std=self.std,
                                       batch_size=self.batch_size,
                                       shuffle=False, num_workers=self.num_workers)
        since = time.time()

        headers = ['epoch', 'loss', 'score']
        headers.extend(self.labels)
        # stats dataframes
        train_stats = pd.DataFrame(columns=headers)
        val_stats = pd.DataFrame(columns=headers)

        # stats paths
        train_stats_path = save_dir + 'train_stats.csv'
        val_stats_path = save_dir + 'val_stats.csv'

        # best_fitness_from_train = 0.0
        # checkpoint = {'epoch': start_epoch,
        #              'model_state_dict': self.model.state_dict(),
        #              'optimizer_state_dict': optimizer.state_dict(),
        #              'loss': 0.0,
        #              'score': 0.0}
        best_fitness_from_val = 0.0
        val_score = 0.0
        val_loss = 0.0
        the_last_loss = 100
        patience = patience
        trigger_times = 0
        best_val = {'epoch': self.start_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': 0.0,
                    'score': 0.0}

        print(('\n' + '%10s' * 3) % ('Epoch', 'loss', 'score'))
        for epoch in range(self.start_epoch, self.epochs):  # epoch -----------------------------------------

            self.model.train()
            running_loss = 0.0

            preds = pd.DataFrame(columns=self.labels)
            trues = pd.DataFrame(columns=self.labels)
            # progres bar
            nb = len(train_loader)
            pbar = tqdm(train_loader, total=nb, ncols=NCOLS,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            # batch ----------------------------------------------------------------------------
            for i, (tag_id, inputs, labels) in enumerate(pbar, 1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    if self.is_inception:
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = self.model(inputs)
                        loss1 = self.criterion(outputs, labels)
                        loss2 = self.criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    loss.backward()
                    self.optimizer.step()

                preds = preds.append(pd.DataFrame(outputs.tolist(), columns=self.labels))
                trues = trues.append(pd.DataFrame(labels.tolist(), columns=self.labels))

                running_loss += loss.item()
                score, labels_score = score_function(preds, trues)

                # score, _ = score_function(pd.DataFrame(outputs.tolist(), columns=self.labels),
                #                          pd.DataFrame(labels.tolist(), columns=self.labels))

                # bar update
                pbar.set_description(
                    ('%10s' * 1 + '%10.4g' * 2) % (f'{epoch + 1}/{self.epochs}', running_loss / i, score))
            # end batch ---------------------------------------------------------------------------

            score, labels_score = score_function(preds, trues)
            epoch_loss = running_loss / len(train_loader)

            # zapisywanie statystyk -------------------------------------------
            epoch_stats = [epoch + 1, epoch_loss, score]
            epoch_stats.extend(labels_score)
            train_stats = train_stats.append(pd.DataFrame([epoch_stats], columns=headers), ignore_index=True)
            train_stats.to_csv(path_or_buf=train_stats_path, index=False)

            final_epoch = (epoch + 1 == self.epochs)
            # walidacja -------------------------------------------
            noval = ((epoch + 1) % self.val_every != 0)
            if not noval or final_epoch:
                val_loss, val_score, val_labels_score = self.val(val_loader=val_loader,
                                                                 score_function=score_function,
                                                                 criterion=self.criterion)
                val_epoch_stats = [epoch + 1, val_loss, val_score]
                val_epoch_stats.extend(val_labels_score)
                val_stats = val_stats.append(pd.DataFrame([val_epoch_stats], columns=headers), ignore_index=True)
                val_stats.to_csv(path_or_buf=val_stats_path, index=False)
            if val_score > best_fitness_from_val:
                best_val = {'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'loss': val_loss,
                            'score': val_score}
                best_fitness_from_val = val_score
                print("New best fitness!", end=" ")
                torch.save(best_val, save_dir + "bestfitness_from_validation.pth")
                print("Saved in " + save_dir + "bestfitness_from_validation.pth")

            # autosave -------------------------------------------
            nosave = ((epoch + 1) % self.save_every != 0)
            if not nosave or final_epoch:  # if save
                print("Autosave...", end=" ")
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': epoch_loss,
                            'score': score}, save_dir + "autosave.pth")
                print("saved in " + save_dir + "autosave.pth")

            # early stopping ----------------------------
            if not noval:
                if val_loss > the_last_loss:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print('Early stopping! No progress')
                        time_elapsed = time.time() - since
                        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                        return self.model, train_stats, val_stats, best_val

                else:
                    trigger_times = 0
                    print("trigger :0")

                the_last_loss = val_loss
            # early stopping ---------------------------
        # end epoch ------------------------------------------------------------------------------

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model, train_stats, val_stats, best_val

    def val(self, val_loader, score_function, criterion=nn.BCELoss()):

        if not callable(score_function):
            raise ValueError("Score should be function")

        self.model.eval()
        running_loss = 0.0

        preds = pd.DataFrame(columns=self.labels)
        trues = pd.DataFrame(columns=self.labels)

        nb = len(val_loader)
        pbar = tqdm(val_loader, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for i, (tag_id, inputs, labels) in enumerate(pbar, 1):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

            preds = preds.append(pd.DataFrame(outputs.tolist(), columns=self.labels))
            trues = trues.append(pd.DataFrame(labels.tolist(), columns=self.labels))

            running_loss += loss.item()
            score, labels_score = score_function(preds, trues)

            # score, _ = score_function(pd.DataFrame(outputs.tolist(), columns=self.labels),
            #                          pd.DataFrame(labels.tolist(), columns=self.labels))
            # bar update
            pbar.set_description(('%10s' + '%10.4g' * 2) % ("Val", running_loss / i, score))

        score, labels_score = score_function(preds, trues)
        epoch_loss = running_loss / len(val_loader)

        return epoch_loss, score, labels_score

    def __call__(self,
               test_df: pd.DataFrame,
               data_dir: str,
               save_dir: str,
               silent_mode=False
               ) -> pd.DataFrame:
        """

        :return: Obiekt z odpowiedziami modelu na testowy zbiór danych
        :rtype: pd.DataFrame
        """
        if test_df.empty:
            raise ValueError("DataFrame cannot be empty")

        if not os.path.exists(data_dir):
            raise ValueError("Data dir doesn't exist")

        dataloader = create_dataloader(data_dir=data_dir, dataframe=test_df, labels=self.labels,
                                       preprocessing=self.preprocessing, augmentation=None,
                                       input_size=self.input_size, mean=self.mean, std=self.std,
                                       batch_size=self.batch_size,
                                       shuffle=False, num_workers=self.num_workers)

        answer = pd.DataFrame(columns=self.labels)

        if not silent_mode:
            nb = len(dataloader)
            pbar = tqdm(dataloader, desc="Detecting", total=nb, ncols=NCOLS,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

            with torch.no_grad():
                self.model.eval()
                for i, (tag_id, inputs, labels) in enumerate(pbar, 0):
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    answer = answer.append(pd.DataFrame(outputs.tolist(), columns=self.labels, index=tag_id.tolist()))
        else:
            with torch.no_grad():
                self.model.eval()
                for tag_id, inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    answer = answer.append(pd.DataFrame(outputs.tolist(), columns=self.labels, index=tag_id.tolist()))

        answer.index.name = 'tag_id'
        answer.to_csv(path_or_buf=save_dir + "answer.csv")
        return answer

    def resume(self,
               path,
               train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               score_function,
               data_dir: str, ):

        self.load_checkpoint(path=path)

        self.train(data_dir, train_df, val_df, score_function)
