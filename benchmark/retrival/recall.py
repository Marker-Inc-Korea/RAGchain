from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple
import pytrec_eval


class RecallFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        recall = {}
        recall[f"Recall@{k_value}"] = 0.0
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recall.' + str(k_value)})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            recall[f"Recall@{k_value}"] += scores[query_id]["recall_" + str(k_value)]

        recall[f"Recall@{k_value}"] = round(recall[f"Recall@{k_value}"] / len(scores), 5)

        return recall