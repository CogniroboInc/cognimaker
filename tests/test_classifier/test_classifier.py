import os
import unittest
import pickle
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from cognimaker.estimator import BaseClassifier


class MyClassifier(BaseClassifier):

    def get_params(self):
        return {
            'max_depth': 5
        }

    def fit(self, X, y, params):
        model = DecisionTreeClassifier(**params)
        model.fit(X, y)

        return model

    def save_model(self, model):
        with open(os.path.join(self.output_dir, 'model.pkl'), mode='wb') as f:
            pickle.dump(model, f)

    def get_predict_proba(self, model, X):
        predict_proba = model.predict_proba(X)
        return predict_proba[:, 1]

    def get_predict(self, model, X):
        return model.predict(X)


class ClassifierTestCase(unittest.TestCase):

    def setUp(self):
        self.input_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'input_dir')
        self.output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'output_dir')
        self.param_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'parameter.json')

    def test_train(self):
        classifier = MyClassifier(
            input_dir=self.input_dir,
            param_path=self.param_path,
            output_dir=self.output_dir,
            pretrain_model_dir=None
        )
        classifier.train()


if __name__ == '__main__':
    unittest.main(exit=False)
