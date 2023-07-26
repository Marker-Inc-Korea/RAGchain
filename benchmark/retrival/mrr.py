from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple


class MRRFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:

        mrr = {}
        mrr[f"MRR@{k_value}"] = 0.0

        top_hits = {}

        for query_id, doc_scores in results.items():
            top_hits[query_id] = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k_value]

        for query_id in top_hits:
            query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])

        for rank, hit in enumerate(top_hits[query_id][0:k_value]):
            if hit[0] in query_relevant_docs:
                mrr[f"MRR@{k_value}"] += 1.0 / (rank + 1)
                break

        mrr[f"MRR@{k_value}"] = round(mrr[f"MRR@{k_value}"] / len(qrels), 5)

        return mrr
