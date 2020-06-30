from dataclasses import dataclass
from typing import Generic, Callable, TypeVar, List, Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix as _sk_confusion_matrix,
    roc_auc_score,
    roc_curve as _sk_roc_curve,
    f1_score,
)

from ..model.prediction import Prediction
from ..model.classification import BinaryClassificationPrediction


T = TypeVar("T")


@dataclass
class Metric(Generic[T]):
    name: str
    func: Callable[[Prediction], T]
    combine: Callable[[List[List[T]]], T]
    log_value: bool = True

    def __call__(self, prediction: Prediction) -> T:
        return self.func(prediction)  # type: ignore


@dataclass
class BinaryClassificationMetric(Metric[T]):
    func: Callable[[BinaryClassificationPrediction], T]


def _mk_sk_binary_metric_pred(
    name, score_func, combine=np.mean
) -> BinaryClassificationMetric:
    def func(prediction: BinaryClassificationPrediction):
        return score_func(prediction.target, prediction.get_prediction())

    return BinaryClassificationMetric(name, func, combine)


def _mk_sk_binary_metric_prob(
    name, score_func, combine=np.mean
) -> BinaryClassificationMetric:
    def func(prediction: BinaryClassificationPrediction):
        return score_func(prediction.target, prediction.get_probability())

    return BinaryClassificationMetric(name, func, combine)


precision: Metric[float] = _mk_sk_binary_metric_pred("precision", precision_score)


recall: Metric[float] = _mk_sk_binary_metric_pred("recall", recall_score)


accuracy: Metric[float] = _mk_sk_binary_metric_pred("accuracy", accuracy_score)


f1: Metric[float] = _mk_sk_binary_metric_pred("f1", f1_score)


auc: Metric[float] = _mk_sk_binary_metric_pred("auc", roc_auc_score)


@dataclass
class RocCurve:
    fpr: Sequence[float]
    tpr: Sequence[float]

    @staticmethod
    def apply(prediction: BinaryClassificationPrediction):
        fpr, tpr, thresholds = _sk_roc_curve(
            prediction.target, prediction.get_probability()
        )
        return RocCurve(fpr=fpr, tpr=tpr)  # 偽陽性率  # 真陽性率

    @staticmethod
    def combine(values: List[List["RocCurve"]]) -> "RocCurve":
        # 偽陽性率、真陽性率をそれぞれ平均してROCとして算出
        # モデルごとのROC曲線ではX軸の真陽性率の値が異なるため線形補完して合わせる
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for xs in values:
            for roc in xs:
                interp_tpr = np.interp(mean_fpr, roc.fpr, roc.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        return RocCurve(fpr=mean_fpr, tpr=mean_tpr)


roc_curve: Metric[RocCurve] = BinaryClassificationMetric(
    name="roc_curve", func=RocCurve.apply, combine=RocCurve.combine, log_value=False
)


@dataclass
class ConfusionMatrix:
    TN: int
    FP: int
    FN: int
    TP: int

    def __add__(self, other: "ConfusionMatrix"):
        return ConfusionMatrix(
            TN=self.TN + other.TN,
            FP=self.FP + other.FP,
            FN=self.FN + other.FN,
            TP=self.TP + other.TP,
        )

    def __truediv__(self, other: int):
        return ConfusionMatrix(
            TN=round(self.TN / other),
            FP=round(self.FP / other),
            FN=round(self.FN / other),
            TP=round(self.TP / other),
        )

    @staticmethod
    def from_sk(cm) -> "ConfusionMatrix":
        """
        Create from scikit-learn confusion matrix array.
        """
        return ConfusionMatrix(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1])

    @staticmethod
    def apply(prediction: BinaryClassificationPrediction) -> "ConfusionMatrix":
        cm = _sk_confusion_matrix(prediction.target, prediction.get_prediction())
        return ConfusionMatrix.from_sk(cm)

    @staticmethod
    def combine(mats: List[List["ConfusionMatrix"]]) -> "ConfusionMatrix":
        def fold1(xs, op):
            v = xs[0]
            t = xs[1:]
            for e in t:
                v = op(v, e)
            return v

        n = len(mats)
        return fold1([cm for xs in mats for cm in xs], ConfusionMatrix.__add__) / n


confusion_matrix: Metric[ConfusionMatrix] = BinaryClassificationMetric(
    name="confusion_matrix",
    func=ConfusionMatrix.apply,
    combine=ConfusionMatrix.combine,
)

BINARY_CLASSIFICATION_METRICS = [
    accuracy,
    precision,
    recall,
    f1,
    auc,
    roc_curve,
    confusion_matrix
]
