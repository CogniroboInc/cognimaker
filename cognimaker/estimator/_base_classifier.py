import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)
from abc import abstractmethod

from ._base import BaseEstimator


class BaseClassifier(BaseEstimator):
    """
    クラス分類モデル（２値分類）の基底クラス
    """
    def calc_indicators(self, model, X, y):
        """
        以下の指標を算出し、self.indicatorsに格納する
        正解率、適合率、再現率、混同行列、AUC、ROC、
        """
        # 予測確率と予測結果の算出
        predict_proba = self.get_predict_proba(model, X)
        y_pred = self.get_predict(model, X)

        # 正解率
        accuracy = accuracy_score(y, y_pred)
        self.indicators["accuracy"] = accuracy
        self.logger.info("accuracy={:.4f};".format(accuracy))

        # 適合率
        precision = precision_score(y, y_pred)
        self.indicators["precision"] = precision
        self.logger.info("precision={:.4f};".format(precision))

        # 再現率
        recall = recall_score(y, y_pred)
        self.indicators["recall"] = recall
        self.logger.info("recall={:.4f};".format(recall))

        # 混同行列
        cm = confusion_matrix(y, y_pred)
        self.indicators["confusion_matrix"] = {
            "TN": cm[0][0],
            "FP": cm[0][1],
            "FN": cm[1][0],
            "TP": cm[1][1]
        }
        self.logger.info("confusion_matrix \n {};".format(cm))

        # AUC（教師データに片方のラベルしか無い場合エラーになるので例外処理をいれる）
        try:
            auc = roc_auc_score(y, predict_proba)
            self.indicators["auc"] = auc
            self.logger.info("auc={:.4f};".format(auc))
        except Exception as e:
            self.indicators["auc"] = None
            self.logger.error(str(e))

        # ROC（教師データに片方のラベルしか無い場合エラーになるので例外処理をいれる）
        try:
            fpr, tpr, thresholds = roc_curve(y, predict_proba)
            # ROC曲線は、fprを横軸、tprを縦軸にプロットしたもの
            self.indicators["roc"] = {
                "fpr": fpr, #偽陽性率
                "tpr": tpr #真陽性率
            }
        except Exception as e:
            self.indicators["roc"] = {}
            self.logger.error(str(e))

    @abstractmethod
    def get_predict_proba(self, model, X):
        """
        モデルの予測確率を取得するメソッド
        Args:
            model: 学習済みモデル
            X: 予測用の入力データ
        Returns:
            予測確率の一次元配列（陽性確率のリスト）
        """
        raise NotImplementedError()

    @abstractmethod
    def get_predict(self, model, X):
        """
        モデルの予測結果を取得するメソッド
        Args:
            model: 学習済みモデル
            X: 予測用の入力データ
        Returns:
            予測結果の一次元配列（予測ラベルのリスト）
        """
        raise NotImplementedError()

