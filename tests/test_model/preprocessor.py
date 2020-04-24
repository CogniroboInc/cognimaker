import pandas as pd
import click

from cognimaker.preprocessor import BasePreprocessor


class Preprocessor(BasePreprocessor):
    ID_COLUMN_NAME = '顧客ID'

    category_column_dict = {
        '性別': 'sex',
        '都道府県': 'pref',
        '誕生月': 'birth_month',
        '支払い方法': 'payment_method'
    }

    continuous_column_dict = {
        '年齢': 'age',
        'オプション1': 'option1',
        'オプション2': 'option2',
        'オプション3': 'option3',
        'オプション4': 'option4',
        'オプション5': 'option5',
        'オプション6': 'option6',
        'オプション7': 'option7',
        'オプション8': 'option8',
        'オプション9': 'option9',
        'オプション10': 'option10',
    }

    agg_category_column_dict = {
        '商品名': 'items',
        'キャンペーンコード': 'campaign',
        'オプション11': 'option11',
        'オプション12': 'option12',
        'オプション13': 'option13',
        'オプション14': 'option14',
        'オプション15': 'option15',
        'オプション16': 'option16',
        'オプション17': 'option17',
        'オプション18': 'option18',
        'オプション19': 'option19',
        'オプション20': 'option20',
    }

    def _get_base_data(self, sql_context, columns):
        query_columns = [
            f'`{Preprocessor.ID_COLUMN_NAME}`',
            'MAX(target) AS target',
            'COUNT(*) AS cnt',
            'datediff(MAX(`日付`),MIN(`日付`)) AS term',
        ]
        self.output_columns.append(Preprocessor.ID_COLUMN_NAME)
        if self.purpose in ['train', 'fine_tune']:
            self.output_columns.append('target')
        self.output_columns.append('cnt')

        for column in columns:
            if column in Preprocessor.category_column_dict.keys():
                query_columns.append(f'MAX(`{column}`) AS {Preprocessor.category_column_dict[column]}')

            if column in Preprocessor.continuous_column_dict.keys():
                query_columns.append(f'MAX(`{column}`) AS {Preprocessor.continuous_column_dict[column]}')
                self.output_columns.append(Preprocessor.continuous_column_dict[column])

        if '単価' in columns:
            query_columns.append('MEAN(`単価`) AS mean_price')
            query_columns.append('CAST(MAX(`単価`) AS INTEGER) AS max_price')
            self.output_columns.append('mean_price')
            self.output_columns.append('max_price')

        if '個数' in columns:
            query_columns.append('MEAN(`個数`) AS mean_quantity')
            query_columns.append('CAST(MAX(`個数`) AS INTEGER) AS max_quantity')
            self.output_columns.append('mean_quantity')
            self.output_columns.append('max_quantity')

        if '金額' in columns:
            query_columns.append('MEAN(`金額`) AS mean_amount')
            query_columns.append('CAST(MAX(`金額`) AS INTEGER) AS max_amount')
            self.output_columns.append('mean_amount')
            self.output_columns.append('max_amount')

        df = sql_context.sql("""
        SELECT
            {0}
        FROM
            tran
        GROUP BY
            `{1}`
        """.format(','.join(query_columns), Preprocessor.ID_COLUMN_NAME)).toPandas()

        # カテゴリ型のカラムをラベルエンコーディング
        for column in columns:
            if column in Preprocessor.category_column_dict.keys():
                df = self.label_encoding(df, Preprocessor.category_column_dict[column])

        return df

    def _agg_category(self, sql_context, column):
        agg = sql_context.sql("""
        SELECT
            `{0}`,
            UPPER(`{1}`) AS category,
            COUNT(*) AS cnt
        FROM
            tran
        GROUP BY
            1, 2
        """.format(Preprocessor.ID_COLUMN_NAME, column))
        df = agg.groupBy(Preprocessor.ID_COLUMN_NAME).pivot('category').sum('cnt').fillna(0).toPandas()
        cols = [column + '_' + col if col != Preprocessor.ID_COLUMN_NAME else col for col in df.columns]
        df.columns = cols
        if self.purpose == 'train':
            self.encoder_dict[column + '_agg'] = {}
            self.encoder_dict[column + '_agg']['columns'] = df.columns
        else:
            base_df = pd.DataFrame(index=[], columns=self.encoder_dict[column + '_agg']['columns'])

            df = pd.concat([base_df, df], axis=0)

            if base_df.shape[1] < df.shape[1]:
                raise ValueError(f'カラム：{column} に学習時に無いカテゴリ値が含まれています。')

        df = df.fillna(0)
        cols = [col for col in df.columns if col != Preprocessor.ID_COLUMN_NAME]
        self.output_columns = self.output_columns + cols
        return df

    def transform(self, spark_df, sql_context):
        columns = spark_df.columns
        spark_df.createOrReplaceTempView("tran")

        df = self._get_base_data(sql_context, columns)

        for column in columns:
            if column in Preprocessor.agg_category_column_dict.keys():
                df_agg = self._agg_category(sql_context, column)
                df = pd.merge(df, df_agg, on=Preprocessor.ID_COLUMN_NAME, how='left')

        return df