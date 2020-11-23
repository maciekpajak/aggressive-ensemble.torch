import json
import torch
import pandas as pd

from aggressive_ensemble.Model import Model


class Ensemble:

    def __init__(self, config_file, ensemble_config):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        config = json.load(open(config_file))
        self.directories = config["directories"]
        self.mode = config["modes"]
        self.files = config["files"]

        self.labels = list(pd.read_csv(self.files["labels_csv"]))

        self.ensemble_config = json.load(open(ensemble_config))

        self.models = self.__load_models()

    def __call__(self):

        if self.mode["train"]:
            self.__train()

        if self.mode["save_models"]:
            print('Saving')
            for model in self.models:
                self.models[model].save()

        if self.mode["test"]:
            answer = self.__test()
            self.__save_answer(answer)

    def __load_models(self):
        print('Loading models in progress')
        csv = {'test': self.files["test_csv"],
               'train': self.files["train_csv"]}
        dir = {'stats': self.directories["stats_dir"],
               'data': self.directories["data_dir"],
               'models': self.directories["models_dir"]}
        return {model: Model(csv, dir, self.labels, self.ensemble_config[model], self.device) for model in
                self.ensemble_config}

    def __train(self):
        print("Training...")
        for model in self.models:
            self.models[model].show_random_images(10)
            self.models[model].train()

    def __test(self):
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

        tmp = answers[0].copy(deep=False).sort_index(ascending=True)

        answers2 = []
        for a in answers:
            answers2.append(a.reset_index(drop=True))

        answers = []
        for a in answers2:
            t = tmp.copy(deep=False)
            for col in range(a.shape[1]):
                t.iloc[:, col] = pd.DataFrame(a.iloc[:, col]).sort_values(by=col, ascending=True).index
            answers.append(t.copy(deep=True))

        for col in range(tmp.shape[1]):
            for row in range(tmp.shape[0]):
                rank = 0.0
                for a in answers:
                    rank += a.iloc[row, col]
                rank = rank / len(answers)
                tmp.iloc[row, col] = rank

        rpreds = tmp.copy(deep=False)
        for col in range(tmp.shape[1]):
            rpreds.iloc[:, col] = tmp.sort_values(by=col, ascending=True).index
        rpreds = rpreds.reset_index(drop=True)
        return rpreds
