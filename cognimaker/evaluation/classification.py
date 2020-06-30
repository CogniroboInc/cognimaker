from dataclasses import dataclass
from typing import Type, Dict
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from .evaluator import Evaluator, EvaluationResult
from ..model.model import SupervisedModel
from ..model.prediction import SupervisedPrediction
from ..util.const import RANDOM_SEED


def group(n, xs):
    return [xs[i:i+n] for i in range(0, len(xs), n)]


@dataclass
class CrossValidationEvaluator(Evaluator):
    model_cls: Type[SupervisedModel]
    num_splits: int = 5
    num_repeats: int = 1

    def run(self, data: pd.DataFrame, model_params: dict) -> EvaluationResult:
        group_splits = partial(group, self.num_splits)
        split_indicators: Dict[str, list] = {
            m.name: [] for m in self.metrics
        }
        cv = RepeatedStratifiedKFold(
            n_splits=self.num_splits,
            n_repeats=self.num_repeats,
            random_state=RANDOM_SEED,
        )
        X, y = self.model_cls.get_feature_target(data)
        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            model = self.model_cls(model_params)
            model.fit(X_train, y_train)
            prediction: SupervisedPrediction = model.predict(X_test)
            prediction = prediction.set_target(y_test)

            for m in self.metrics:
                result = m(prediction)
                split_indicators[m.name].append(result)

        combined_indicators = {
            m.name: m.combine(group_splits(split_indicators[m.name]))  # type: ignore
            for m in self.metrics
        }
        model = self.model_cls(model_params)
        model.fit(X, y)
        result = EvaluationResult(model, combined_indicators)
        return result


@dataclass
class ValidationSplitEvaluator(Evaluator):
    model_cls: Type[SupervisedModel]
    test_size: float = 0.2

    def run(self, data: pd.DataFrame, model_params: dict) -> EvaluationResult:
        X, y = self.model_cls.get_feature_target(data)

        X_train, y_train, X_test, y_test = \
            train_test_split(X, y, test_size=self.test_size, random_state=RANDOM_SEED)

        model = self.model_cls(model_params)
        model.fit(X_train, y_train)
        prediction: SupervisedPrediction = model.predict(X_test)
        prediction = prediction.set_target(y_test)

        combined_indicators = {
            m.name: m(prediction) for m in self.metrics
        }
        model = self.model_cls(model_params)
        model.fit(X, y)
        result = EvaluationResult(model, combined_indicators)
        return result
