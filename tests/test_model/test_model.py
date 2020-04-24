import os
import unittest
import pickle
import numpy as np
import pandas as pd
import lightgbm

from .preprocessor import Preprocessor
from .dropout_predictor import DropoutPredictor


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        self.input_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'input')
        self.output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'parameter.json')
        self.categorical_columns = [
            '性別','都道府県','誕生月','支払い方法','商品名','キャンペーンコード',
            'オプション11','オプション12','オプション13','オプション14','オプション15',
            'オプション16','オプション17','オプション18','オプション19','オプション20',
        ]

    def test_0_train_preprocess(self):
        input_path = self.input_dir + '/train.csv'
        output_path = self.input_dir + '/train/preprocessed.csv'
        pickle_path = self.input_dir + '/train_preprocess.pkl'
        purpose = 'train'
        load_pickle_path = None

        preprocessor = Preprocessor(
            input_path=input_path,
            output_path=output_path,
            pickle_path=pickle_path,
            purpose=purpose,
            load_pickle_path=load_pickle_path,
            categorical_columns=self.categorical_columns
        )
        preprocessor.preprocess()

    def test_1_train_model(self):
        input_dir = self.input_dir + '/train'
        param_path = self.input_dir + '/parameter.json'
        save_model_dir = self.output_dir + '/model'
        model = DropoutPredictor(
            input_dir=input_dir,
            param_path=param_path,
            save_model_dir=save_model_dir,
            pretrain_model_dir=None
        )
        model.train()

    def test_2_test_preprocess(self):
        input_path = self.input_dir + '/test.csv'
        output_path = self.input_dir + '/test/preprocessed.csv'
        pickle_path = self.input_dir + '/test_preprocess.pkl'
        purpose = 'predict'
        load_pickle_path = self.input_dir + '/train_preprocess.pkl'

        preprocessor = Preprocessor(
            input_path=input_path,
            output_path=output_path,
            pickle_path=pickle_path,
            purpose=purpose,
            load_pickle_path=load_pickle_path,
            categorical_columns=self.categorical_columns
        )
        preprocessor.preprocess()

    def test_3_test_predict(self):
        model_path = self.output_dir + '/model/model.txt'
        input_path =  self.input_dir + '/test/preprocessed.csv'
        output_path = self.output_dir + '/test/prediction.csv'
        df = pd.read_csv(input_path, header=None)
        ids = list(df.iloc[:, 0])

        data = df.iloc[:, 1:]

        predict_proba = model.predict(input)
        prediction = np.round(predict_proba).astype(int)

        pd.DataFrame(
            {'id': ids, 'prediction': predictions, 'probability': probability},
            columns=['prediction', 'probability']
        ).to_csv(output_path, header=True, index=False)


if __name__ == '__main__':
    unittest.main(exit=False)