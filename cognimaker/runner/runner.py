from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..model.model import Model
from ..evaluation.evaluator import Evaluator


class ModelConfig(ABC):
    @abstractmethod
    def get_default_params(self) -> dict:
        pass

    @abstractmethod
    def get_score_metric(self) -> str:
        """Return the name of the metric to use when optimizing hyperparameters"""
        pass

    @abstractmethod
    def create_model(self, model_params: dict) -> Model:
        pass

    @abstractmethod
    def load_model(self, model_dir: Path) -> Model:
        pass

    @abstractmethod
    def get_evaluator(self, data: pd.DataFrame) -> Evaluator:
        pass


class Runner:
    pass
