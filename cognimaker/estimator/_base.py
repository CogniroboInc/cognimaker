import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod
from ..util import get_logger, NumpyEncoder


class BaseEstimator(ABC):

    INDICATOR_FILE = 'indicator.json'
    RANDOM_SEED = 592 #Cogni
    SCORE_FORMAT = "MODEL_SCORE={};"

    def __init__(self, input_dir: str, output_dir: str, param_path: str = None, pretrain_model_dir: str = None):
        """
        param
            input_dir: 学習データの保存先ディレクトリのパス
            output_dir: モデルファイルのや指標の出力先ディレクトリのパス
            param_path: 学習パラメータの設定ファイルのパス
            pretrain_model_dir: 学習済みモデルファイルの保存先ディレクトリのパス
        """
        self.input_dir = input_dir
        self.param_path = param_path
        self.output_dir = output_dir
        self.pretrain_model_dir = pretrain_model_dir
        self.process_id = self._get_process_id()
        self.logger = get_logger(self.__class__.__name__, self.process_id)
        self.indicators = {}

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
            self.logger.info(f"data size: {len(X)}")
            test_size = self.get_test_size()
            X_train, X_test, y_train, y_test = \
                train_test_split(
                    X, y, test_size=test_size, random_state=self.RANDOM_SEED)
            model = self.fit(X_train, y_train, params)
            self.save_model(model)
            self.log_score(model, X_test, y_test)
            self.calc_indicators(model, X_test, y_test)
            self.save_indicators()
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
        raise NotImplementedError()

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
        raw_data = [pd.read_csv(file, header=0) for file in input_files]
        data = pd.concat(raw_data)
        # 学習データの形式は以下を想定している
        # １列目　　：IDカラム
        # ２列目　　：教師カラム
        # ３列目以降：特徴量カラム
        y = data.iloc[:, 1].values
        X = data.iloc[:, 2:].values

        # 特徴量のカラム名を取得
        self.feature_columns = data.columns[2:]

        return X, y

    def get_test_size(self) -> float:
        """
        訓練用データ（train）と検証用データ（test）の比率を決める関数
        """
        return 0.2

    def log_score(self, model, X, y):
        """
        モデルのスコアを算出し、標準出力に出力するメソッド
        trainメソッド内で呼び出される。

        Args:
            model: 学習ずみのモデルインスタンス
            X: スコア算出用の特徴量データ
            y: スコア算出用の教師データ
        """
        score = self.get_score(model, X, y)
        self.logger.info(self.SCORE_FORMAT.format(score))
        # indicatorに追加
        self.indicators["score"] = score

    @abstractmethod
    def get_score(self, model, X, y) -> float:
        """
        モデルのスコアを算出するメソッド
        モデルごとの評価指標をインプリメントする
        単一のスコア指標を返す。

        Args:
            model: 学習ずみのモデルインスタンス
            X: スコア算出用の特徴量データ
            y: スコア算出用の教師データ
        return
            score: モデルのスコア指標
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    @abstractmethod
    def calc_indicators(self, model, X, y) -> None:
        """
        モデルの評価指標を算出するメソッド
        Args:
            model: 学習済みのモデルインスタンス
            X: 指標算出用の入力データ
            y: 指標算出用の教師データ
        """
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, model) -> None:
        """
        モデルを保存するメソッド（継承したクラスで実装する必要あり）
        output_dirで指定したディレクトリにモデルファイルを保存する
        保存するファイルの形式やファイル名はメソッド内で記述する
        param
            model: 保存するモデルのインスタンス
        """
        raise NotImplementedError()

    def save_indicators(self):
        """
        モデルの評価指標をJSON形式で出力するメソッド
        output_dirで指定したディレクトリに保存する
        """
        with open(os.path.join(self.output_dir, self.INDICATOR_FILE), 'w') as f:
            json.dump(self.indicators, f, cls=NumpyEncoder, indent=2)
