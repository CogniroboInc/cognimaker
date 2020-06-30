from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Optional

import pandas as pd

from .model import UnsupervisedModel
from .prediction import Prediction


@dataclass
class RecommendationPrediction(Prediction):
    truth: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self._validate()

    def set_truth(self, truth: pd.DataFrame):
        return replace(self, truth=truth)

    def _validate(self):
        assert('user' in self.prediction.columns)
        assert('item' in self.prediction.columns)
        assert('score' in self.prediction.columns)
        assert('rank' in self.prediction.columns)

        #from pandas.testing import assert_index_equal
        if self.truth is not None:
            #assert_index_equal(self.prediction.index, self.target.index)
            pass


class RecommendationModel(UnsupervisedModel):
    @classmethod
    def get_user_item_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[['user', 'item', 'timestamp']]

    @classmethod
    def get_user_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        cols = ['user'] + list(c for c in data.columns if c.startswith('user_'))
        return (
            data
            [cols]
            .drop_duplicates()
            .set_index('user')
        )

    @classmethod
    def get_item_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        cols = ['item'] + list(c for c in data.columns if c.startswith('item_'))
        return (
            data
            [cols]
            .drop_duplicates()
            .set_index('item')
        )

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, users: pd.Series) -> RecommendationPrediction:
        pass
