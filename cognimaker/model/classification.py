from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .model import SupervisedModel
from .prediction import SupervisedPrediction, DataFrameOrSeries


@dataclass
class BinaryClassificationPrediction(SupervisedPrediction):
    target: Optional[pd.Series] = None

    def _validate(self):
        super()._validate()

        if self.target is not None:
            assert(self.target.min() >= 0.0 and self.target.max() <= 1.0)

        predcols = set(self.prediction.columns)
        assert('prediction' in predcols and predcols <= {'prediction', 'probability'})

    def get_probability(self) -> Optional[pd.Series]:
        if 'probability' in self.prediction:
            return self.prediction['probability']
        else:
            return None

    def get_prediction(self) -> pd.Series:
        return self.prediction['prediction']


class BinaryClassificationModel(SupervisedModel):
    @classmethod
    def get_feature_target(
        cls, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, DataFrameOrSeries]:
        return data.iloc[:, 1:], data.iloc[:, 0]

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> BinaryClassificationPrediction:
        pass
