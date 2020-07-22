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
    def calc_indicators(self, model, X, y) -> dict:
        """
        モデルの評価指標を算出するメソッド

        以下の指標を算出し返す
        正解率、適合率、再現率、混同行列、AUC、ROC、

        Args:
            model: 学習済みのモデルインスタンス
            X: 指標算出用の入力データ
            y: 指標算出用の教師データ
        return
            indicators: インジケーターを格納した辞書オブジェクト
        """
        # 予測確率と予測結果の算出
        predict_proba = self.get_predict_proba(model, X)
        y_pred = self.get_predict(model, X)

        indicators = {}

        # 正解率
        accuracy = accuracy_score(y, y_pred)
        indicators["accuracy"] = accuracy
        self.logger.info("accuracy={:.4f};".format(accuracy))

        # 適合率
        precision = precision_score(y, y_pred)
        indicators["precision"] = precision
        self.logger.info("precision={:.4f};".format(precision))

        # 再現率
        recall = recall_score(y, y_pred)
        indicators["recall"] = recall
        self.logger.info("recall={:.4f};".format(recall))

        # 混同行列
        cm = confusion_matrix(y, y_pred)
        indicators["confusion_matrix"] = {
            "TN": cm[0][0],
            "FP": cm[0][1],
            "FN": cm[1][0],
            "TP": cm[1][1]
        }
        self.logger.info("confusion_matrix \n {};".format(cm))

        # AUC（教師データに片方のラベルしか無い場合エラーになるので例外処理をいれる）
        try:
            auc = roc_auc_score(y, predict_proba)
            indicators["auc"] = auc
            self.logger.info("auc={:.4f};".format(auc))
        except Exception as e:
            indicators["auc"] = None
            self.logger.error(str(e))

        # ROC（教師データに片方のラベルしか無い場合エラーになるので例外処理をいれる）
        try:
            fpr, tpr, thresholds = roc_curve(y, predict_proba)
            # ROC曲線は、fprを横軸、tprを縦軸にプロットしたもの
            indicators["roc"] = {
                "fpr": fpr, #偽陽性率
                "tpr": tpr #真陽性率
            }
        except Exception as e:
            indicators["roc"] = {}
            self.logger.error(str(e))

        return indicators

    def log_indicators(self) -> None:
        """
        ログのために標準出力にモデルの評価指標を出力するメソッド
        """
        # 正解率
        accuracy = self.indicators["accuracy"] 
        self.logger.info("accuracy={:.4f};".format(accuracy))

        # 適合率
        precision = self.indicators["precision"] 
        self.logger.info("precision={:.4f};".format(precision))

        # 再現率
        recall = self.indicators["recall"] 
        self.logger.info("recall={:.4f};".format(recall))

        # 混同行列
        cm = self.indicators["confusion_matrix"]
        self.logger.info("confusion_matrix \n {};".format(cm))

        # AUC（教師データに片方のラベルしか無い場合エラーになるので例外処理をいれる）
        auc = self.indicators["auc"] 
        self.logger.info("auc={:.4f};".format(auc))

    def combine_indicators(self, indicators: list) -> dict:
        """
        クロスバリデーションの場合に複数のモデルの評価指標を一つに集約するメソッド

        基本的にはそれぞれの指標の値の平均を算出する
        """

        indicator_count = len(indicators)

        # 正解率
        accuracy = 0
        for ind in indicators:
            accuracy = accuracy + ind["accuracy"] 
        accuracy = accuracy / indicator_count

        # 適合率
        precision = 0
        for ind in indicators:
            precision = precision + ind["precision"] 
        precision = precision / indicator_count

        # 再現率
        recall = 0
        for ind in indicators:
            recall = recall + ind["recall"] 
        recall = recall / indicator_count

        # 混同行列
        cm = { "TN": 0, "FP": 0, "FN": 0, "TP": 0 }
        for ind in indicators:
            cm["TN"] = cm["TN"] + ind["confusion_matrix"]["TN"]
            cm["FP"] = cm["FP"] + ind["confusion_matrix"]["FP"]
            cm["FN"] = cm["FN"] + ind["confusion_matrix"]["FN"]
            cm["TP"] = cm["TP"] + ind["confusion_matrix"]["TP"]
        cm["TN"] = cm["TN"] / indicator_count
        cm["FP"] = cm["FP"] / indicator_count
        cm["FN"] = cm["FN"] / indicator_count
        cm["TP"] = cm["TP"] / indicator_count
        

        # AUC（教師データに片方のラベルしか無い場合エラーになるので例外処理をいれる）
        auc = 0
        auc_cnt = indicator_count
        for ind in indicators:
            if ind["auc"] is not None:
                auc = auc + ind["auc"] 
            else:
                auc_cnt = auc_cnt - 1
        auc = auc / auc_cnt

        # 偽陽性率、真陽性率をそれぞれ平均してROCとして算出
        # モデルごとのROC曲線ではX軸の真陽性率の値が異なるため線形補完して合わせる
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for ind in indicators:
            if "fpr" in ind["roc"]:
                interp_tpr = np.interp(mean_fpr, ind["roc"]["fpr"], ind["roc"]["tpr"])
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        roc = {
            "fpr": mean_fpr,
            "tpr": mean_tpr
        }

        result = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "auc": auc,
            "roc": roc
        }

        return result

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

