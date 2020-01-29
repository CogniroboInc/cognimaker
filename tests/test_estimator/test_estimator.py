import os
import unittest
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from cognimaker.estimator import BaseEstimator


class Estimator(BaseEstimator):

    def get_params(self):
        return {
            'max_depth': 5
        }

    def fit(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        # output model scores
        self.log_model_score(model, X_train, y_train, 'train')
        self.log_model_score(model, X_val, y_val, 'valid')

    def save_model(self, model):
        with open(os.path.join(self.save_model_dir, 'model.pkl'), mode='wb') as f:
            pickle.dump(model, f)

    def log_model_score(self, model, X, y, score_type='train'):
        predict_proba = model.predict(X)
        prediction = np.round(predict_proba)

        accuracy = accuracy_score(y, prediction)
        precision = precision_score(y, prediction)
        recall = recall_score(y, prediction)
        auc = roc_auc_score(y, predict_proba)
        f1 = f1_score(y, prediction)

        self.log(level='info', message=score_type + "_accuracy={:.4f};".format(accuracy))
        self.log(level='info', message=score_type + "_precision={:.4f};".format(precision))
        self.log(level='info', message=score_type + "_recall={:.4f};".format(recall))
        self.log(level='info', message=score_type + "_auc={:.4f};".format(auc))
        self.log(level='info', message=score_type + "_f1={:.4f};".format(f1))


class EstimatorTestCase(unittest.TestCase):

    def setUp(self):
        self.input_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'input_dir')
        self.save_model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'save_model_dir')
        self.param_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'parameter.json')

    def test_train(self):
        estimator = Estimator(
            input_dir=self.input_dir,
            param_path=self.param_path,
            save_model_dir=self.save_model_dir,
            pretrain_model_dir=None
        )
        estimator.train()


if __name__ == '__main__':
    unittest.main(exit=False)
