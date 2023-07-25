from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple
import pytrec_eval


class NDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        ndcg = {}
        ndcg[f"NDCG@{k_value}"] = 0.0
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndgc_cut.' + str(k_value)})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            ndcg[f"NDCG@{k_value}"] += scores[query_id]["ndcg_cut_" + str(k_value)]

        ndcg[f"NDCG@{k_value}"] = round(ndcg[f"NDCG@{k_value}"] / len(scores), 5)

        return ndcg