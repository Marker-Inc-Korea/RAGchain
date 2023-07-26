from benchmark.retrival.base_retrival_evaluation_factory import BaseRetrievalEvaluationFactory
from typing import List, Dict, Union, Tuple

class HoleFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        hole = {}
        hole[f"Hole@{k_value}"] = 0.0

        annotated_corpus = set()
        for _, docs in qrels.items():
            for doc_id, score in docs.items():
                annotated_corpus.add(doc_id)

        for _, doc_scores in results.items():
            top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_value]

            hole_docs = [row[0] for row in top_hits[0:k_value] if row[0] not in annotated_corpus]
            hole[f"Hole@{k_value}"] += len(hole_docs) / k_value

        hole[f"Hole@{k_value}"] = round(hole[f"Hole@{k_value}"] / len(qrels), 5)


        return hole