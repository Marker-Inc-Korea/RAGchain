from abc import ABC, abstractmethod
from typing import Dict
import math
from operator import itemgetter

class BaseAnswerMetric(ABC):
    def __init__(self):
        self._metric_name = None

    @property
    def metric_name(self):
        return str(self._metric_name)

    def eval(self, solution: str,
             pred: str) -> float:
        metric = self.retrieval_metric_function(solution, pred)

        return metric

    def _normalizer_str(self, s: str) -> str:
        return s.lower().strip()

    @abstractmethod
    def retrieval_metric_function(self, solution: str,
                                  pred: str) -> float:
        pass