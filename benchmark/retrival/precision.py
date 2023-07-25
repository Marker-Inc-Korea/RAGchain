from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple
import pytrec_eval


class PrecisionFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        precision = {}
        precision[f"Precision@{k_value}"] = 0.0
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'P.' + str(k_value)})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            precision[f"Precision@{k_value}"] += scores[query_id]["P_" + str(k_value)]

        precision[f"Precision@{k_value}"] = round(precision[f"Precision@{k_value}"] / len(scores), 5)

        return precision