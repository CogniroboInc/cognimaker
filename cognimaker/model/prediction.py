from dataclasses import dataclass, replace
from typing import Optional, Union

import pandas as pd


DataFrameOrSeries = Union[pd.DataFrame, pd.Series]


@dataclass
class Prediction:
    """
    Container for a prediction from a model.
    """
    prediction: DataFrameOrSeries


@dataclass
class SupervisedPrediction(Prediction):
    target: Optional[DataFrameOrSeries] = None

    def __post_init__(self):
        self._validate()

    def set_target(self, target: DataFrameOrSeries) -> "SupervisedPrediction":
        return replace(self, target=target)

    def _validate(self) -> None:
        from pandas.testing import assert_index_equal
        if self.target is not None:
            assert_index_equal(self.prediction.index, self.target.index)
