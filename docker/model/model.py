from pathlib import Path

import pandas as pd
import numpy as np

from cognimaker.runner import ModelConfig
from cognimaker.model.classification import (
    BinaryClassificationModel,
    BinaryClassificationPrediction,
)
from cognimaker.evaluation.evaluator import Evaluator
from cognimaker.evaluation.classification import (
    CrossValidationEvaluator,
    ValidationSplitEvaluator,
)
from cognimaker.evaluation.metric import BINARY_CLASSIFICATION_METRICS
from cognimaker.model.prediction import DataFrameOrSeries

from sklearn.ensemble import RandomForestClassifier


DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": 10}


class TestModel(BinaryClassificationModel):
    def fit(self, X: pd.DataFrame, y: DataFrameOrSeries) -> None:
        self.model = RandomForestClassifier(**self.params).fit(X, y)

    def predict(self, data: pd.DataFrame) -> BinaryClassificationPrediction:
        prob = self.model.predict_proba(data)[:, 1]
        pred = np.round(prob)
        return BinaryClassificationPrediction(
            pd.DataFrame({"prediction": pred, "probability": prob}, index=data.index)
        )


class Config(ModelConfig):
    def get_default_params(self) -> dict:
        return DEFAULT_PARAMS

    def get_score_metric(self) -> str:
        """Return the name of the metric to use when optimizing hyperparameters"""
        return "f1"

    def create_model(self, model_params: dict) -> TestModel:
        return TestModel()

    def load_model(self, model_dir: Path) -> TestModel:
        return TestModel.load(model_dir)

    def get_evaluator(self, data: pd.DataFrame) -> Evaluator:
        if data.size > 5000:
            return ValidationSplitEvaluator(
                TestModel, BINARY_CLASSIFICATION_METRICS, test_size=0.25
            )
        else:
            return CrossValidationEvaluator(
                TestModel, BINARY_CLASSIFICATION_METRICS, num_splits=5, num_repeats=2
            )


config = Config()
