from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple
import pytrec_eval


class MAPFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        _map = {}
        _map[f"MAP@{k_value}"] = 0.0
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map_cut.' + str(k_value)})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            _map[f"MAP@{k_value}"] += scores[query_id]["map_cut_" + str(k_value)]

        _map[f"MAP@{k_value}"] = round(_map[f"MAP@{k_value}"] / len(scores), 5)

        return _map