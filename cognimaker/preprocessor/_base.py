import os
import pickle
import boto3
import pandas as pd
import numpy as np

from logging import getLogger, StreamHandler, Formatter, INFO
from io import StringIO
from abc import ABC, abstractmethod
from pyspark import SparkContext
from pyspark.sql import SQLContext
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from ..util import get_logger


class BasePreprocessor(ABC):

    # loggerはpickle化できないためクラス変数として保持する。
    logger = get_logger(__name__)

    @staticmethod
    def _is_s3_path(path: str):
        return path.startswith('s3://')

    @staticmethod
    def _parse_s3_file_path(s3_file_path):
        """
        S3ファイルパスS3バケット名とS3キーパスを取得
        :param s3_path:
        :return bucket(str), key(str)
        """
        bucket = s3_file_path.split('//')[1].split('/')[0]
        key = s3_file_path.split(f's3://{bucket}/')[1]
        return bucket, key

    def __init__(
        self, input_path: str, output_path: str, pickle_path: str,
        purpose: str = 'train', load_pickle_path: str = None
    ):
        """
        param
            input_path: 前処理前の入力ファイルのパス
            input_path: 前処理完了後の出力ファイルのパス
            pickle_path: pickleを保存（又はロード）するパス
            purpose: 前処理の目的（train, predict, fine_tune）を選択
            load_pickle_path: 訓練時のオブジェクトを保存したpickleファイルのパス
        """
        self.process_id = os.environ.get('PROCESS_ID', 'xxxxxxxx')

        if purpose not in ["train", "predict", "fine_tune"]:
            self.logger.error('invalid purpose')
            raise ValueError('invalid purpose')

        if purpose in ["predict", "fine_tune"]:
            if BasePreprocessor._is_s3_path(load_pickle_path):
                bucket, key = BasePreprocessor._parse_s3_file_path(load_pickle_path)
                s3 = boto3.resource('s3')
                response = s3.Object(bucket, key).get()
                obj = pickle.loads(response['Body'].read())
                self.encoder_dict = obj.encoder_dict
            else:
                with open(load_pickle_path, mode='rb') as f:
                    obj = pickle.load(f)
                    self.encoder_dict = obj.encoder_dict

        else:
            # カラムとエンコーダーの対応関係を管理するdict
            self.encoder_dict = {}

        # 前処理結果として出力するカラム
        self.output_columns = []
        self.purpose = purpose
        self.input_path = input_path
        self.output_path = output_path
        self.pickle_path = pickle_path

    def _to_pickle(self):
        if BasePreprocessor._is_s3_path(self.pickle_path):
            bucket, key = BasePreprocessor._parse_s3_file_path(self.pickle_path)
            s3 = boto3.resource('s3')
            s3.Object(bucket, key).put(Body=pickle.dumps(self))
        else:
            with open(self.pickle_path, mode='wb') as f:
                pickle.dump(self, f)

    def _load_data(self):
        """
        CSVファイルをspark data frameに読み込むメソッド
        param
            file: 前処理を行うデータファイルのパス（S3）
        return
            spark_data_frame:
        """
        spark_context = SparkContext.getOrCreate()
        sql_context = SQLContext(spark_context)
        spark_data_frame = sql_context.read.format('com.databricks.spark.csv') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load(self.input_path)

        return spark_data_frame, spark_context, sql_context

    @abstractmethod
    def transform(self, spark_df, sql_context) -> pd.DataFrame:
        """
        モデル固有の前処理を実装するメソッド
        param
            spark_df: 処理対象データを読み込んだspark DataFrame
            sql_context: 前処理を行うsparkのsql_context
        return
            df: 前処理が完了した後のpandas DataFrame
        """
        pass

    def _output(self, pandas_df):
        """
        前処理が完了したDataFrameをS3に出力するメソッド
        param
            pandas_df: 出力するpandas DataFrame
        """
        if BasePreprocessor._is_s3_path(self.output_path):
            buffer = StringIO()
            pandas_df[self.output_columns].to_csv(buffer, index=False, header=False)
            output_bucket, output_key = BasePreprocessor._parse_s3_file_path(self.output_path)
            s3 = boto3.resource('s3')
            s3.Object(output_bucket, output_key).put(Body=buffer.getvalue())
        else:
            pandas_df.to_csv(self.output_path, index=False, header=False)

    def preprocess(self):
        """
        前処理の一連の流れを実行するメソッド
        データ読み込み→前処理→出力
        の一連の流れを行う
        このクラスを継承したクラスで、transformメソッドを実装する必要がある
        """
        try:
            self.logger.info("start preprocess")
            # データ読み込み
            spark_df, spark_context, sql_context = self._load_data()
            self.logger.info("complete load data")
            # 前処理
            pandas_df = self.transform(spark_df, sql_context)
            self.logger.info("complete transform")
            # 出力
            self._output(pandas_df)
            self.logger.info("complete output")
            # インスタンスを保存
            self._to_pickle()
            self.logger.info("complete save instance")
            # spoark sessionの終了
            spark_context.stop()
            self.logger.info("complete preprocess")
        except Exception as e:
            self.logger.error(str(e))
            raise e

    def one_hot_encoding(self, df, column) -> pd.DataFrame:
        """
        指定したカラムをOneHotEncodingするメソッド
        param
            column: OneHotEncodingを行う対象カラム名
            df: 処理を行うpandas.DataFrame
        return
            df: OneHotEncodingを行った結果をカラムに追加したpandas.DataFrame
        """
        if self.purpose == 'train':
            le = LabelEncoder()
            labels = le.fit_transform(df[column])

            ohe = OneHotEncoder()
            encoded = ohe.fit_transform(labels.reshape(-1, 1)).astype(int)

            self.encoder_dict[column] = {}
            self.encoder_dict[column]['label_encoder'] = le
            self.encoder_dict[column]['one_hot_encoder'] = ohe
        else:
            le = self.encoder_dict[column]['label_encoder']
            ohe = self.encoder_dict[column]['one_hot_encoder']

            labels = le.transform(df[column])
            encoded = ohe.transform(labels.reshape(-1, 1)).astype(int)

        names = [(column + "-") + str(s) for s in le.classes_]

        one_hot_df = pd.DataFrame(
            index=df.index,
            columns=names,
            data=encoded.toarray(),
            dtype=int
        )
        df = pd.concat([df, one_hot_df], axis=1)
        self.output_columns.extend(names)

        return df

    def label_encoding(self, df, column) -> pd.DataFrame:
        """
        指定したカラムに対して、LabelEncodingを行うメソッド
        param
            column: LabelEncodingを行う対象カラム名
            df: 処理を行うpandas.DataFrame
        return
            df: LabelEncodingを行った結果をカラムに追加したpandas.DataFrame
        """
        encode_column_name = column + '_label'

        if self.purpose == 'train':
            le = LabelEncoder()
            df[encode_column_name] = le.fit_transform(df[column])
            self.encoder_dict[encode_column_name] = {}
            self.encoder_dict[encode_column_name]['label_encoder'] = le
        else:
            le = self.encoder_dict[encode_column_name]['label_encoder']
            df[encode_column_name] = le.transform(df[column])

        self.output_columns.append(encode_column_name)
        return df

    def frequency_encoding(self, df, column) -> pd.DataFrame:
        """
        指定したカラムに対して、FrequencyEncodingを行うメソッド
        param
            column: FrequencyEncodingを行う対象カラム名
            df: 処理を行うpandas.DataFrame
        return
            df: FrequencyEncodingを行った結果をカラムに追加したpandas.DataFrame
        """
        encode_column_name = column + '_freq'

        if self.purpose == 'train':
            freq = df[column].value_counts()
            self.encoder_dict[encode_column_name] = {}
            self.encoder_dict[encode_column_name]['freq'] = freq
        else:
            freq = self.encoder_dict[encode_column_name]['freq']

        df[encode_column_name] = df[column].map(freq)

        self.output_columns.append(encode_column_name)
        return df

    def target_encoding(
        self,
        df,
        column,
        target_column=None,
        random_state=0,
        n_splits=4
    ) -> pd.DataFrame:
        """
        指定したカラムに対して、TargetEncodingを行うメソッド
        param
            column: TargetEncodingを行う対象カラム名
            df: 処理を行うpandas.DataFrame
            target_column: 目的変数のカラム名
            random_state: KFoldでデータを分割する時の乱数シード
            n_splits: KFoldでデータを分割する時の分割数
        return
            df: TargetEncodingを行った結果をカラムに追加したpandas.DataFrame
        """
        encode_column_name = column + '_target_encode'

        if self.purpose == 'train':
            # 学習データの変換後の値を格納する配列を準備
            tmp = np.repeat(np.nan, df.shape[0])

            # 学習データを分割
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            for idx_1, idx_2 in kf.split(df):
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean = df.iloc[idx_1].groupby(column)[target_column].mean()
                # 変換後の値を一次配列に格納
                tmp[idx_2] = df[column].iloc[idx_2].map(target_mean)

            # 変換後のデータをカラムに格納
            df[encode_column_name] = tmp

            target_mean = df.groupby(column)[target_column].mean()
            self.encoder_dict[encode_column_name] = {}
            self.encoder_dict[encode_column_name]['target_mean'] = target_mean
        else:
            target_mean = self.encoder_dict[encode_column_name]['target_mean']
            df[encode_column_name] = df[column].map(target_mean)

        self.output_columns.append(encode_column_name)
        return df
