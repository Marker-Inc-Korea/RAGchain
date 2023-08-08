from beir.retrieval import evaluation
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
import pytrec_eval
import json


class BaseRetrievalEvaluationFactory(ABC):

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

class NDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
                                  results: Dict[str, Dict[str, float]],
                                  k_value: int = 1) -> Dict[str, float]:
        ndcg = {}
        ndcg[f"NDCG@{k_value}"] = 0.0
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.' + str(k_value)})
        scores = evaluator.evaluate(results)
        for query_id in scores.keys():
            ndcg[f"NDCG@{k_value}"] += scores[query_id]["ndcg_cut_" + str(k_value)]

        ndcg[f"NDCG@{k_value}"] = round(ndcg[f"NDCG@{k_value}"] / len(scores), 5)

        return ndcg

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

# class RecallCapFactory(BaseRetrievalEvaluationFactory):
#     def retrieval_metric_function(self, qrels: Dict[str, Dict[str, int]],
#                                   results: Dict[str, Dict[str, float]],
#                                   k_value: int = 1) -> Dict[str, float]:
#         capped_recall = {}
#         capped_recall[f"Recall_cap@{k_value}"] = 0.0
#
#         for query_id, doc_scores in results.items():
#             top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_value]
#             query_relevant_docs = [doc_id for doc_id, doc_score in qrels[query_id].items() if doc_score > 0]
#             retrieved_docs = [row[0] for row in top_hits[0:k_value] if qrels[query_id].get(row[0], 0) > 0]
#             denominator = min(len(query_relevant_docs), k_value)
#             capped_recall[f"Recall_cap@{k_value}"] += (len(retrieved_docs) / denominator)
#
#         capped_recall[f"Recall_cap@{k_value}"] = round(capped_recall[f"Recall_cap@{k_value}"] / len(qrels), 5)
#
#         return capped_recall

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