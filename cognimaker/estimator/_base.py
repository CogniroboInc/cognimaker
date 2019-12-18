import os
import pandas as pd

from abc import ABC, abstractmethod


class BaseEstimator(ABC):

    def __init__(self, input_dir: str, param_path: str, save_model_dir: str, pretrain_model_dir: str = None):
        self.input_dir = input_dir
        self.param_path = param_path
        self.save_model_dir = save_model_dir
        self.pretrain_model_dir = pretrain_model_dir

    def train(self):
        params = self.get_params()
        print(params)
        X, y = self.get_data()
        model = self.fit(X, y, params)
        self.save_model(model)

    @abstractmethod
    def get_params(self) -> dict:
        pass

    def get_data(self) -> (pd.DataFrame, pd.Series):
        # Take the set of files and read them all into a single pandas dataframe
        input_files = [os.path.join(self.input_dir, file) for file in os.listdir(self.input_dir)]
        if len(input_files) == 0:
            raise ValueError(
                (
                    'There are no files in {}.\n'
                    'the data specification in S3 was incorrectly specified or\n'
                    'the role specified does not have permission to access the data.'
                ).format(self.input_dir)
            )
        raw_data = [pd.read_csv(file) for file in input_files]
        data = pd.concat(raw_data)
        # 学習データの形式は以下を想定している
        # １列目　　：IDカラム
        # ２列目　　：教師カラム
        # ３列目以降：特徴量カラム
        y = data.iloc[:, 1]
        X = data.iloc[:, 2:]

        return X, y

    @abstractmethod
    def fit(self, X, y, params):
        pass

    @abstractmethod
    def save_model(self, model):
        pass
