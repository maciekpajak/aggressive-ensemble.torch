import json
import torch
import pandas as pd

from aggressive_ensemble.Model import Model


class Ensemble:

    def __init__(self, config_file, ensemble_config):

        #if config_file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        config = json.load(open(config_file))
        self.directories = config["directories"]
        self.mode = config["modes"]
        self.files = config["files"]

        self.labels = list(pd.read_csv(self.files["labels_csv"]))

        self.ensemble = json.load(open(ensemble_config))

        self.models = self.__load_models()

    def __call__(self):

        if self.mode["train"]:
            self.train()

        if self.mode["save_models"]:
            print('Saving')
            for model in self.models:
                self.models[model].save()

        if self.mode["test"]:
            answer = self.test()
            self.__save_answer(answer)

    def __load_models(self):
        print('Loading models in progress')
        csv = {'test': self.files["test_csv"],
               'train': self.files["train_csv"]}
        dir = {'stats': self.directories["stats_dir"],
               'data': self.directories["data_dir"],
               'models': self.directories["models_dir"]}
        return {model: Model(csv, dir, self.labels, self.ensemble[model], self.device) for model in
                self.ensemble}

    def train(self):
        print("Training...")
        for model in self.models:
            self.models[model].show_random_images(5)
            self.models[model].train()
            self.models[model].save()

    def test(self):
        print('Testing...')
        answers = {model: self.models[model].test() for model in self.models}
        return self.combine_answers(answers)

    def __save_answer(self, answer):

        answer.to_csv(path_or_buf=self.files["answer_csv"], index=False, header=self.labels)
        print('Answer saved as ' + self.files["answer_csv"])

    def __save_model(self, model, model_path):
        pass

    @staticmethod
    def combine_answers(answer):

        print('Combining answers...')

        answers = list(answer.values())

        tags = answers[0].iloc[:, 0].values
        tmp = answers[0].copy(deep=False).set_index(tags)

        for col in tmp.columns.values:
            for tag in tags:
                rank = 0.0
                for ans in answers:
                    rank += ans[ans[col] == tag].index.values[0]
                tmp.loc[tag, col] = rank / len(answers)

        rpreds = tmp.copy(deep=False)
        for col in tmp.columns.values:
            rpreds.loc[:, col] = tmp.sort_values(by=col, ascending=True).index
        rpreds = rpreds.reset_index(drop=True)

        return rpreds
