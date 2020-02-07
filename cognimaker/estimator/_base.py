import os
import json
import pandas as pd

from abc import ABC, abstractmethod
from ..util import get_logger


class BaseEstimator(ABC):

    def __init__(self, input_dir: str, save_model_dir: str, param_path: str = None, pretrain_model_dir: str = None):
        """
        param
            input_dir: 学習データの保存先ディレクトリのパス
            save_model_dir: モデルファイルの保存先ディレクトリのパス
            param_path: 学習パラメータの設定ファイルのパス
            pretrain_model_dir: 学習済みモデルファイルの保存先ディレクトリのパス
        """
        self.input_dir = input_dir
        self.param_path = param_path
        self.save_model_dir = save_model_dir
        self.pretrain_model_dir = pretrain_model_dir
        self.process_id = self._get_process_id()
        self.logger = get_logger(self.__class__.__name__, self.process_id)

    def _get_process_id(self):
        with open(self.param_path, 'r') as tc:
            params = json.load(tc)
        process_id = params.get('process_id', 'xxxxxxxx')
        return process_id

    def train(self) -> None:
        """
        学習の一連の処理を実行するメソッド
        パラメータの取得→学習データの読み込み→学習→モデルの保存
        の一連の流れを行う
        """
        try:
            self.logger.info("start training")
            params = self.get_params()
            self.logger.info(json.dumps(params))
            X, y = self.get_data()
            model = self.fit(X, y, params)
            self.save_model(model)
            self.logger.info("complete training")
        except Exception as e:
            self.logger.error(str(e))
            raise e

    @abstractmethod
    def get_params(self) -> dict:
        """
        学習モデルに与えるパラメータを取得するメソッド（継承したクラスで実装する必要あり）
        param_pathで与えたJSONファイルから取得する想定
        指定された値がファイルに無い場合は、別途デフォルト値を設定する
        return
            params: モデルに与えるパラメータのdict
        """
        pass

    def get_data(self) -> (pd.DataFrame, pd.Series):
        """
        学習データを読み込むメソッド
        input_dirで指定したフォルダにあるcsvファイルを読み込み
        特徴量データ(X)と教師データ(y)を返す
        return
            X: 学習用の特徴量データ（pandas.DataFrame）
            y: 学習用の教師データ（pandas.Series）
        """
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
    def fit(self, X, y, params) -> object:
        """
        学習を実行するメソッド（継承したクラスで実装する必要あり）
        学習のアルゴリズムの実装やモデルの評価指標の出力などは、
        このメソッドの中で実装する
        param
            X: 学習用の特徴量データ（pandas.DataFrame）
            y: 学習用の教師データ（pandas.Series）
            params: モデルのハイパーパラメータ（dict）
        return
            model: モデルのインスタンス
        """
        pass

    @abstractmethod
    def save_model(self, model) -> None:
        """
        モデルを保存するメソッド（継承したクラスで実装する必要あり）
        save_model_dirで指定したディレクトリにモデルファイルを保存する
        保存するファイルの形式やファイル名はメソッド内で記述する
        param
            model: 保存するモデルのインスタンス
        """
        pass
