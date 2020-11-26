import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

import pandas as pd

import time
import pkbar

from sklearn.metrics import average_precision_score
import numpy as np

from .Models.init_model import init_model
from .Dataset.ImageDataset import ImageDataset
from .Tranforms.TransformImage import TransformImage


class Model:

    def __init__(self, csv, dir, labels, model_config, device):

        self.device = device

        self.train_csv = csv['train']
        self.test_csv = csv['test']
        self.data_dir = dir['data']
        self.stats_dir = dir['stats']
        self.models_dir = dir['models']

        self.model_config = model_config

        self.labels = labels

        self.model = self.__load_model()

        self.criterion = self.set_criterion(model_config["criterion"])
        self.optimizer = optim.SGD(self.__get_params_to_update(),
                                   lr=model_config["lr"],
                                   momentum=model_config["momentum"])

        self.dataset = self.__create_datasets()

        self.dataloader = {"train": DataLoader(self.dataset["train"],
                                               batch_size=model_config["batch_size"],
                                               shuffle=False,
                                               num_workers=model_config["num_workers"]),
                           "val": DataLoader(self.dataset["val"],
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=model_config["num_workers"]),
                           "test": DataLoader(self.dataset["test"],
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=model_config["num_workers"])}

        self.is_inception = (model_config["name"] == 'inception')

        self.model_id = self.__create_id()

        print("Model %s succesfully initialized" % (self.model_config['name']))

    def __load_model(self):

        if self.model_config["path"] != "":
            model = torch.load(self.model_config['path'], map_location=torch.device(self.device))
        else:
            name = self.model_config["name"]
            num_classes = len(self.labels)
            feature_exctract = self.model_config["feature_extract"]
            pretrained = self.model_config['pretrained']
            model, input_size, mean, std = init_model(name, num_classes, feature_exctract, use_pretrained=pretrained)
            self.model_config["input_size"] = input_size
            self.model_config['preprocessing']['normalization']['mean'] = mean
            self.model_config['preprocessing']['normalization']['std'] = std
            model.to(self.device)
        print('Model ' + self.model_config['name'] + ' loaded')
        return model

    def __create_id(self):
        preprocessing = self.model_config['preprocessing']
        augmentation = self.model_config['augmentation']

        t = 'pretrained=%s' % self.model_config['pretrained']

        mean = preprocessing['normalization']['mean']
        std = preprocessing['normalization']['std']
        p = 'preproc=('
        if preprocessing["polygon_extraction"]:
            p += "PolExtr,"
        if preprocessing["ratation_to_horizontal"]:
            p += "RotToH,"
        if preprocessing["edge_detection"]:
            p += "Edge,"
        if preprocessing["RGB_to_HSV"]:
            p += "HSV,"
        if preprocessing['normalization']:
            p += "Norm=mean%s,std%s" % (mean, std)
        if p == 'preproc=(':
            p += 'None'
        p += ')'

        a = 'aug='
        if augmentation["random_vflip"]:
            a += "RandVFlip,"
        if augmentation["random_hflip"]:
            a += "RandHFlip,"
        if augmentation["random_rotation"]:
            a += "RandRot,"
        if augmentation["switch_RGB_channel"]:
            a += "ChannelSwitch,"
        if a == '':
            a += 'None'

        save_name = self.model_config['name'] + '_' + self.model_config['criterion'] + '_' + p + '_' + a + '_' + t
        print(save_name)
        return save_name

    def __get_params_to_update(self):

        params_to_update = self.model.parameters()
        print("Params to learn:")
        if self.model_config["feature_extract"]:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print("\t", name)
        return params_to_update

    def __create_datasets(self, ratio=0.8):

        train_transform = TransformImage(input_size=self.model_config['input_size'],
                                         mean=self.model_config['preprocessing']['normalization']['mean'],
                                         std=self.model_config['preprocessing']['normalization']['std'],
                                         preprocessing=self.model_config['preprocessing'],
                                         augmentations=self.model_config['augmentation'])

        test_transform = TransformImage(input_size=self.model_config['input_size'],
                                        mean=self.model_config['preprocessing']['normalization']['mean'],
                                        std=self.model_config['preprocessing']['normalization']['std'],
                                        preprocessing=self.model_config['preprocessing'],
                                        augmentations=None)

        train_dataset = ImageDataset(csv_file=self.train_csv,
                                     data_dir=self.data_dir,
                                     labels=self.labels,
                                     transform=train_transform)

        val_dataset = ImageDataset(csv_file=self.train_csv,
                                   data_dir=self.data_dir,
                                   labels=self.labels,
                                   transform=test_transform)

        test_dataset = ImageDataset(csv_file=self.test_csv,
                                    data_dir=self.data_dir,
                                    labels=self.labels,
                                    transform=test_transform)

        l = int(len(train_dataset) * ratio)
        train_dataset = Subset(train_dataset, range(0, l))
        val_dataset = Subset(val_dataset, range(l, len(val_dataset)))

        dataset = {"train": train_dataset,
                   "val": val_dataset,
                   "test": test_dataset}

        return dataset

    def save(self):
        torch.save(self.model, self.models_dir + self.model_id + '.pth')

    def train(self):

        since = time.time()

        max_epochs = self.model_config['max_epochs']
        val_every = self.model_config['val_every']

        headers = ['epoch', 'loss', 'score']
        headers.extend(self.labels)
        stats = {'train': pd.DataFrame(columns=headers),
                 'val': pd.DataFrame(columns=headers)}
        stats_path = {'train': self.stats_dir + self.model_id + '_train_stats.csv',
                      'val': self.stats_dir + self.model_id + '_val_stats.csv'}

        epoch_score_history = [0, 0, 0]
        for epoch in range(max_epochs):

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    bar = pkbar.Kbar(target=len(self.dataloader[phase]), epoch=epoch, num_epochs=max_epochs, width=50,
                                     always_stateful=True)
                else:
                    if (epoch + 1) % val_every != 0:
                        break
                    self.model.eval()
                    bar = pkbar.Kbar(target=len(self.dataloader[phase]), width=50,
                                     always_stateful=True)

                running_loss = 0.0
                preds = torch.empty(0, 37).to(self.device)
                trues = torch.empty(0, 37).to(self.device)
                for i, (tag_id, inputs, labels) in enumerate(self.dataloader[phase], 0):

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    preds = torch.cat((preds, outputs), dim=0)
                    trues = torch.cat((trues, labels), dim=0)

                    running_loss += loss.item()
                    score, _ = self.score(outputs, labels)
                    # bar update
                    bar.update(i, values=[("loss", loss.item()), ('score', score)])

                score, labels_score = self.score(preds, trues)
                epoch_loss = running_loss * inputs.size(0) / len(self.dataloader[phase].dataset)

                bar.add(1, values=[("epoch_loss", epoch_loss), ('epoch_score', score)])

                epoch_stats = [epoch + 1, epoch_loss, score]
                epoch_stats.extend(labels_score)
                stats[phase] = stats[phase].append(pd.DataFrame([epoch_stats], columns=headers), ignore_index=True)
                stats[phase].to_csv(path_or_buf=stats_path[phase], index=False)

                if phase == 'train':
                    epoch_score_history[epoch % 3] = score

            if epoch_score_history[0] == epoch_score_history[1] and epoch_score_history[1] == epoch_score_history[2]:
                break
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model

    def test(self):

        preds = []
        tags = []
        bar = pkbar.Kbar(target=len(self.dataloader['test']), width=50, always_stateful=True)
        with torch.no_grad():
            self.model.eval()
            for i, (tag_id, inputs, labels) in enumerate(self.dataloader['test'], 0):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds.append(outputs.tolist()[0])
                tags.append(tag_id.item())
                bar.update(i)
            bar.add(1)

        answer = pd.DataFrame(preds, index=tags).astype(float)
        answer = self.rank_preds(answer)
        print(answer)
        return answer

    @staticmethod
    def set_criterion(criterion_name):
        if criterion_name == 'BCE':
            criterion = nn.BCELoss()
        elif criterion_name == 'BCEL':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        return criterion

    @staticmethod
    def rank_preds(preds):
        rpreds = pd.DataFrame(preds)
        for col in range(preds.shape[1]):
            rpreds.iloc[:, col] = preds.sort_values(by=col, ascending=False).index
        return rpreds

    def show_random_images(self, num):
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

    @staticmethod
    def score(preds, trues):

        labels_score = []
        trues = trues.transpose(0, 1)
        preds = preds.transpose(0, 1)

        trues = trues.to('cpu').detach().numpy()
        preds = preds.to('cpu').detach().numpy()

        score = 0.0
        for p, t in zip(preds, trues):
            ap = 0.0
            if np.sum(t) != 0:
                ap = average_precision_score(t, p)
            labels_score.append(ap)
            score += ap

        score /= preds.shape[0]

        return score, labels_score
