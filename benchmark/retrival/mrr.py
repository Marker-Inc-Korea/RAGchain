from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple


class MRRFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:

        mrr = {}
        mrr[f"MRR@{k_value}"] = 0.0
        mrr[f"MRR@{k_value}"] = round(mrr[f"Recall@{k_value}"] / len(scores), 5)

        return mrr