from beir.retrieval import evaluation
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
import pytrec_eval
import json


class BaseRetrivalEvaluationFactory(ABC):

    def eval(self, qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k: int) -> List[Dict[str, float]]:

        assert k > 0, "k must be greater than 0"
        metric = self.retrieval_metric_function(qrels, results, k)

        return metric

    @abstractmethod
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        pass
