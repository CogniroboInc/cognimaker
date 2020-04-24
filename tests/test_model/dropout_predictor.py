import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)
import lightgbm
from cognimaker.estimator import BaseEstimator


class DropoutPredictor(BaseEstimator):

    def get_params(self):
        # Read in any hyperparameters that
        # the user passed with the training job
        with open(self.param_path, 'r') as tc:
            params = json.load(tc)

        param_settings = [
            ['boosting_type', 'gbdt', str],
            ['class_weight', None, str],
            ['colsample_bytree', 1.0, float],
            ['learning_rate', 0.1, float],
            ['max_depth', -1, int],
            ['min_child_samples', 20, int],
            ['min_child_weight', 0.001, float],
            ['min_split_gain', 0.0, float],
            ['n_estimators', 100, int],
            ['num_leaves', 31, int],
            ['random_state', None, int],
            ['reg_alpha', 0.0, float],
            ['reg_lambda', 0.0, float],
            ['subsample', 1.0, float],
            ['objective', 'binary', str],
            ['metric', 'auc', str],
            ['is_finetune', False, bool],
            ['num_boost_round', 100, int],
            ['early_stopping_rounds', 10, int],
        ]

        for key, default, value_type in param_settings:
            if default is None:
                params[key] = params.get(key, default)
                if params[key] is not None:
                    params[key] = value_type(params[key])
            else:
                params[key] = value_type(params.get(key, default))

        return params

    def fit(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)
        lgb_train = lightgbm.Dataset(X_train, y_train, free_raw_data=False)
        lgb_val = lightgbm.Dataset(X_val, y_val, free_raw_data=False)
        num_boost_round = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        if params['is_finetune']:
            pretrain_model_file_path = os.path.join(
                self.pretrain_model_dir, os.listdir(
                    self.pretrain_model_dir)[0]
            )
            model = lightgbm.train(
                params,
                lgb_train,
                init_model=pretrain_model_file_path,
                valid_sets=[lgb_val, lgb_train],
                valid_names=['val', 'train'],
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True
            )
        else:
            model = lightgbm.train(
                params,
                lgb_train,
                valid_sets=[lgb_val, lgb_train],
                valid_names=['val', 'train'],
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True
            )

        # output model scores
        try:
            self.log_model_score(model, X_train, y_train, 'train')
            self.log_model_score(model, X_val, y_val, 'valid')
        except Exception as e:
            self.logger.error(str(e))

        return model

    def log_model_score(self, model, X, y, score_type='train'):
        predict_proba = model.predict(X)
        prediction = np.round(predict_proba)

        accuracy = accuracy_score(y, prediction)
        precision = precision_score(y, prediction)
        recall = recall_score(y, prediction)
        auc = roc_auc_score(y, predict_proba)
        f1 = f1_score(y, prediction)

        self.logger.info(score_type + "_accuracy={:.4f};".format(accuracy))
        self.logger.info(score_type + "_precision={:.4f};".format(precision))
        self.logger.info(score_type + "_recall={:.4f};".format(recall))
        self.logger.info(score_type + "_auc={:.4f};".format(auc))
        self.logger.info(score_type + "_f1={:.4f};".format(f1))

    def save_model(self, model):
        model.save_model(os.path.join(self.save_model_dir, 'model.txt'))
