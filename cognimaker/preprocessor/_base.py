import pickle
import boto3
import pyspark.sql.dataframe

from io import StringIO
from pyspark import SparkContext
from pyspark.sql import SQLContext


class BasePreprocessor(object):

    def __init__(self, input_path: str, output_path: str, purpose: str = 'train'):
        """
        param input_path: 前処理前の入力ファイルのパス
        param input_path: 前処理完了後の出力ファイルのパス
        param purpose: 前処理の目的（train, predict, fine_tune）を選択
        """
        self.input_path = input_path
        self.output_path = output_path
        if purpose not in ["train", "predict", "fine_tune"]:
            raise ValueError('invalid purpose')
        else:
            self.purpose = purpose
        self.encoder_dict = {}

    def to_pickle(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    def load_data(self):
        """
        CSVファイルをspark data frameに読み込む
        :param file:
        :return spark_data_frame:
        """
        spark_context = SparkContext()
        sql_context = SQLContext(spark_context)
        spark_data_frame = sql_context.read.format('com.databricks.spark.csv') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load(self.input_path)

        return spark_data_frame, sql_context

    def transform(self, spark_df, sql_context):
        raise NotImplementedError()

    def output(self, pandas_df):
        buffer = StringIO()
        pandas_df.to_csv(buffer, index=False, header=False)
        output_bucket = self.output_path.split('//')[1].split('/')[0]
        output_key = self.output_path.split(f's3://{output_bucket}/')[1]
        s3 = boto3.resource('s3')
        s3.Object(output_bucket, output_key).put(Body=buffer.getvalue())

    def preprocess(self):
        spark_df, sql_context = self.load_data()
        pandas_df = self.transform(spark_df, sql_context)
        self.output(pandas_df)
