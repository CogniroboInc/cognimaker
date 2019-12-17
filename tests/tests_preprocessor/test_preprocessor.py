import unittest
import pandas as pd

from cognimaker.preprocessor import BasePreprocessor


class Preprocessor(BasePreprocessor):

    def _get_base_data(self, sql_context, columns):
        query_columns = [
            'ID',
            'MAX(target) AS target',
            'COUNT(*) AS cnt',
            'datediff(MAX(`購買日`),MIN(`購買日`)) AS term',
        ]
        self.output_columns.append('ID')
        if self.purpose in ['train', 'fine_tune']:
            self.output_columns.append('target')
        self.output_columns.append('cnt')

        if '性別' in columns:
            query_columns.append('MAX(`性別`) AS sex')
        if '年齢' in columns:
            query_columns.append('MAX(`年齢`) AS age')
            self.output_columns.append('age')
        if '年代' in columns:
            query_columns.append('MAX(`年代`) AS age_class')

        if '合計金額' in columns:
            query_columns.append('MEAN(`合計金額`) AS mean_amount')
            query_columns.append('CAST(MAX(`合計金額`) AS INTEGER) AS max_amount')
            self.output_columns.append('mean_amount')
            self.output_columns.append('max_amount')

        df = sql_context.sql("""
        SELECT
            {0}
        FROM
            tran
        GROUP BY
            ID
        """.format(','.join(query_columns))).toPandas()

        if '性別' in columns:
            df = self.label_encoding(df, 'sex')

        if '年代' in columns:
            df = self.one_hot_encoding(df, 'age_class')

        return df

    def _agg_category(self, sql_context, column):
        agg = sql_context.sql("""
        SELECT
            ID,
            UPPER(`{0}`) AS category,
            COUNT(*) AS cnt
        FROM
            tran
        GROUP BY
            1, 2
        """.format(column))
        df = agg.groupBy('ID').pivot('category').sum('cnt').fillna(0).toPandas()
        cols = [column + '_' + col if col != 'ID' else col for col in df.columns]
        df.columns = cols
        if self.purpose == 'train':
            self.encoder_dict[column + '_agg'] = {}
            self.encoder_dict[column + '_agg']['columns'] = df.columns
            return df
        else:
            base_df = pd.DataFrame(index=[], columns=self.encoder_dict[column + '_agg']['columns'])

            df = pd.concat([base_df, df], axis=0)

        df = df.fillna(0)
        cols = [col for col in df.columns if col != 'ID']
        self.output_columns = self.output_columns + cols
        return df

    def transform(self, spark_df, sql_context):
        columns = spark_df.columns
        spark_df.createOrReplaceTempView("tran")

        df = self._get_base_data(sql_context, columns)

        if '販売店' in columns:
            df_store = self._agg_category(sql_context, '販売店')
            df = pd.merge(df, df_store, on='ID', how='left')

        if 'カテゴリ' in columns:
            df_category = self._agg_category(sql_context, 'カテゴリ')
            df = pd.merge(df, df_category, on='ID', how='left')

        return df


class PreprocessorTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_0_transform_train(self):
        preprocessor = Preprocessor(
            input_path="s3a://cognimaker-test/input/test.csv",
            output_path="s3://cognimaker-test/output/test.csv",
            pickle_path="s3://cognimaker-test/pickle/train_preprocessor.pickle",
            purpose="train"
        )

        preprocessor.preprocess()

    def test_1_transform_predict(self):
        preprocessor = Preprocessor(
            input_path="s3a://cognimaker-test/input/test_drop.csv",
            output_path="s3://cognimaker-test/output/test_drop.csv",
            pickle_path="s3://cognimaker-test/pickle/predict_preprocessor.pickle",
            purpose="predict",
            load_pickle_path="s3://cognimaker-test/pickle/train_preprocessor.pickle"
        )

        preprocessor.preprocess()


if __name__ == '__main__':
    unittest.main(exit=False)
