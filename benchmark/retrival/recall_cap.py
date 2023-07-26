from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple


class RecallCapFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        capped_recall = {}
        capped_recall[f"Recall_cap@{k_value}"] = 0.0

        for query_id, doc_scores in results.items():
            top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_value]
            query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]

            retrieved_docs = [row[0] for row in top_hits[0:k_value] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k_value)
            capped_recall[f"R_cap@{k_value}"] += (len(retrieved_docs) / denominator)

        capped_recall[f"Recall_cap@{k_value}"] = round(capped_recall[f"Recall_cap@{k_value}"] / len(qrels), 5)

        return capped_recall
