from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple


class TopKAccuracyFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        top_k_acc = {}
        top_k_acc[f"Accuracy@{k_value}"] = 0.0

        top_hits = {}

        for query_id, doc_scores in results.items():
            top_hits[query_id] = [item[0] for item in
                                  sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_value]]

        for query_id in top_hits:
            query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k_value]:
                    top_k_acc[f"Accuracy@{k_value}"] += 1.0
                    break

        top_k_acc[f"Accuracy@{k_value}"] = round(top_k_acc[f"Accuracy@{k_value}"] / len(qrels), 5)

        return top_k_acc