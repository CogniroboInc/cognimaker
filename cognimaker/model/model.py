from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from .prediction import Prediction, SupervisedPrediction, DataFrameOrSeries


@dataclass  # type: ignore
class Model(ABC):
    params: dict

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Prediction:
        pass

    def save(self, model_dir: Path) -> None:
        import pickle

        with (model_dir / "model.pkl").open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_dir: Path) -> "Model":
        import pickle

        with (model_dir / "model.pkl").open("rb") as f:
            return pickle.load(f)


class SupervisedModel(Model):
    @classmethod
    @abstractmethod
    def get_feature_target(
        cls, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, DataFrameOrSeries]:
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: DataFrameOrSeries) -> None:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> SupervisedPrediction:
        pass


class UnsupervisedModel(Model):
    pass
