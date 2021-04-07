from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, List, Dict, Any

import pandas as pd

from .metric import Metric
from ..model.model import Model


@dataclass
class EvaluationResult:
    model: Model
    indicators: Dict[str, Any]


@dataclass  # type: ignore
class Evaluator(ABC):
    model_cls: Type[Model]
    metrics: List[Metric]

    @abstractmethod
    def run(self, data: pd.DataFrame, model_params: dict) -> EvaluationResult:
        pass
