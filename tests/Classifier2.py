import os
import time
import warnings
import click

import pandas as pd
import torch
from torch import optim, nn

from tqdm import tqdm
import shutil

from src.aggressive_ensemble.utils.general import create_dataloader

NCOLS = shutil.get_terminal_size().columns


class Classifier2:
    """ Klasa reprezentująca pojedynczy model sieci neuronowej.
    Oprócz samego modelu przechowuje wymagane atrybuty

    """

    def __init__(self,
                 name: str,
                 model,
                 device: str,
                 is_inception=False,
                 checkpoint_path=None
                 ):

        self.id = name

        if device not in ["cpu", "gpu"]:
            raise ValueError("Device should be either cpu or gpu")

        self.device = device
        if device == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available! Switched to CPU")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.is_inception = is_inception

        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise ValueError("Checkpoint pathdoesn't exist")
            else:
                self.load_checkpoint(checkpoint_path)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.model

    def load_checkpoint(self, path):

        if not os.path.exists(path):
            raise ValueError("Checkpoint path doesn't exist")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def __get_params_to_update(self):
        """

        :return: Parameters to update
        :rtype: list
        """

        params_to_update = []
        # params_to_update = self.model.parameters()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return params_to_update

    def save(self, path):
        torch.save(self.model, path)

    def train(self,
              save_dir: str,
              score_function,
              train_loader,
              val_loader,
              epochs=50,
              start_epoch=0,
              lr=0.01,
              momentum=0.9,
              val_every=1,
              save_every=3,
              criterion=nn.BCELoss(),
              optimizer_state_dict=None
              ):
        """

        :return: Przetrenowany model
        :rtype:
        """

        if not os.path.exists(save_dir):
            raise ValueError("Save dir doesn't exist")

        if not callable(score_function):
            raise ValueError("Score should be a function")

        since = time.time()

        train_stats = pd.DataFrame()
        val_stats = pd.DataFrame()

        optimizer = optim.SGD(self.__get_params_to_update(), lr=lr,
                              momentum=momentum)
        if optimizer_state_dict is not None:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except Exception as e:
                print(e)

        best_fitness_from_val = 0.0
        val_score = 0.0
        val_loss = 0.0
        the_last_loss = 100
        patience = 2
        trigger_times = 0

        best_val = {'epoch': start_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': 0.0,
                    'score': 0.0}

        print(('\n' + '%10s' * 3) % ('Epoch', 'loss', 'score'))
        for epoch in range(start_epoch, epochs):  # epoch -----------------------------------------

            self.model.train()
            running_loss = 0.0

            preds = pd.DataFrame()
            trues = pd.DataFrame()
            # progres bar
            nb = len(train_loader)
            pbar = tqdm(train_loader, total=nb, ncols=NCOLS,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            # batch ----------------------------------------------------------------------------
            for i, (tag_id, inputs, labels) in enumerate(pbar, 1):
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

                preds = preds.append(pd.DataFrame(outputs.tolist()))
                trues = trues.append(pd.DataFrame(labels.tolist()))

                running_loss += loss.item()
                score, labels_score = score_function(preds, trues)

                # bar update
                pbar.set_description(
                    ('%10s' * 1 + '%10.4g' * 2) % (f'{epoch + 1}/{epochs}', running_loss / i, score))
            # end batch ---------------------------------------------------------------------------

            score, labels_score = score_function(preds, trues)
            epoch_loss = running_loss / len(train_loader)

            # zapisywanie statystyk -------------------------------------------
            epoch_stats = [epoch + 1, epoch_loss, score]
            epoch_stats.extend(labels_score)
            train_stats = train_stats.append(pd.DataFrame([epoch_stats]), ignore_index=True)

            final_epoch = (epoch + 1 == epochs)
            # walidacja -------------------------------------------
            noval = ((epoch + 1) % val_every != 0)
            if not noval or final_epoch:
                val_loss, val_score, val_labels_score = self.val(val_loader=val_loader,
                                                                 score_function=score_function,
                                                                 criterion=criterion)
                val_epoch_stats = [epoch + 1, val_loss, val_score]
                val_epoch_stats.extend(val_labels_score)
                val_stats = val_stats.append(pd.DataFrame([val_epoch_stats]), ignore_index=True)

            if val_score > best_fitness_from_val:
                best_val = {'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}
                best_fitness_from_val = val_score

            # autosave -------------------------------------------
            nosave = ((epoch + 1) % save_every != 0)
            if not nosave or final_epoch:  # if save
                print("Autosave...", end=" ")
                torch.save(best_val, save_dir + "autosave.pth")
                print("saved in " + save_dir + "autosave.pth")

            # early stopping ----------------------------
            if not noval:
                if val_loss > the_last_loss:
                    trigger_times += 1

                    if trigger_times >= patience:
                        print('Early stopping! No progress')
                        return self.model, train_stats, val_stats, best_val

                else:
                    trigger_times = 0

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

        preds = pd.DataFrame()
        trues = pd.DataFrame()

        nb = len(val_loader)
        pbar = tqdm(val_loader, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for i, (tag_id, inputs, labels) in enumerate(pbar, 1):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

            preds = preds.append(pd.DataFrame(outputs.tolist()))
            trues = trues.append(pd.DataFrame(labels.tolist()))

            running_loss += loss.item()
            score, labels_score = score_function(preds, trues)

            # bar update
            pbar.set_description(('%10s' + '%10.4g' * 2) % ("Val", running_loss / i, score))

        score, labels_score = score_function(preds, trues)
        epoch_loss = running_loss / len(val_loader)

        return epoch_loss, score, labels_score

    def __call__(self, dataloader) -> pd.DataFrame:
        """

        :return: Obiekt z odpowiedziami modelu na testowy zbiór danych
        :rtype: pd.DataFrame
        """

        answer = pd.DataFrame()

        nb = len(dataloader)
        pbar = tqdm(dataloader, desc="Detecting", total=nb, ncols=NCOLS,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

        with torch.no_grad():
            self.model.eval()
            for i, (tag_id, inputs, labels) in enumerate(pbar, 0):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                answer = answer.append(pd.DataFrame(outputs.tolist(), index=tag_id.tolist()))

        answer.index.name = 'tag_id'
        return answer

    def resume(self,
               path,
               save_dir: str,
               score_function,
               train_loader,
               val_loader,
               epochs=50,
               start_epoch=0,
               lr=0.01,
               momentum=0.9,
               val_every=1,
               save_every=3,
               criterion=nn.BCELoss()):

        if not os.path.exists(path):
            raise ValueError("Checkpoint pathdoesn't exist")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']

        self.train(save_dir,
                   score_function,
                   train_loader,
                   val_loader,
                   epochs,
                   start_epoch,
                   lr,
                   momentum,
                   val_every,
                   save_every,
                   criterion,
                   optimizer_state_dict=checkpoint['optimizer_state_dict'])
