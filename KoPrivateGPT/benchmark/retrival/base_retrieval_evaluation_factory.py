from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
import pytrec_eval
import json
import math
from operator import itemgetter


class BaseRetrievalEvaluationFactory(ABC):

    def eval(self, solution: Dict[str, int],
             pred: Dict[str, float],
             k: int) -> Dict[str, float]:

        assert k > 0, "k must be greater than 0"
        metric = self.retrieval_metric_function(solution, pred, k)

        return metric

    @abstractmethod
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        pass


class APFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        ap = {}
        ap[f"AP@{k_value}"] = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        count_relevant = 0

        for index, doc_id in enumerate(top_hits):
            if doc_id in query_relevant_docs:
                count_relevant += 1
                precision_at_relevant_doc = count_relevant / (index + 1)
                ap[f"AP@{k_value}"] += precision_at_relevant_doc

        return ap

class NDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        ndcg = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg = sum((2 ** relevance - 1)/ math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.items()[1]))
        dcg = sum((2 ** top_hits_ideal[doc_id] - 1) / math.log2(i + 2) if doc_id in solution else 0
                  for i, doc_id in enumerate(top_hits))

        ndcg[f"NDCG@{k_value}"] += dcg / idcg if idcg > 0 else 0 # ndcg need to deal whole query, so this is not complete ndcg


        return ndcg

class CGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        cg = {}

        cg[f"CG@{k_value}"] = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        cg[f"CG@{k_value}"] += sum(top_hits_ideal[doc_id] if doc_id in solution else 0 for doc_id in top_hits)

        return cg

class IND_DCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        dcg_ind = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        dcg_ind[f"DCG_Ind@{k_value}"] = sum((2 ** top_hits_ideal[doc_id] - 1) / math.log2(i + 2) if doc_id in solution else 0
                  for i, doc_id in enumerate(top_hits))

        return dcg_ind

class DCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        dcg = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        dcg[f"DCG@{k_value}"] = sum(top_hits_ideal[doc_id] / math.log2(i + 2) if doc_id in solution else 0
                  for i, doc_id in enumerate(top_hits))

        return dcg

class IND_IDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        idcg_ind = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg_ind[f"DCG_Ind@{k_value}"] = sum((2 ** relevance - 1)/ math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.items()[1]))

        return idcg_ind

class IDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        idcg = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg[f"IDCG@{k_value}"] = sum(relevance/ math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.items()[1]))

        return idcg

class RecallFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:

        recall = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        assert len(top_hits) > 0, "pred must have at least one document"
        recall[f"Recall@{k_value}"] = len(relevant_retrieved_docs) / len(top_hits)

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
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        precision = {}

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        precision[f"Precision@{k_value}"] = len(relevant_retrieved_docs) / len(query_relevant_docs) if len(query_relevant_docs) > 0 else 0

        return precision

class RRFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        """
         Reciprocal Rank (RR) is the reciprocal of the rank of the first relevant item.
         Mean of RR in whole querys is MRR.
        """
        rr = {}
        rr[f"RR@{k_value}"] = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        for rank, doc_id in enumerate(top_hits):
            if doc_id in query_relevant_docs:
                rr[f"RR@{k_value}"] += 1.0 / (rank + 1)
                break

        return rr

class HoleFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        hole = {}
        hole[f"Hole@{k_value}"] = 0.0

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        hole_docs = [pred_doc_id for pred_doc_id in top_hits if pred_doc_id not in query_relevant_docs]
        hole[f"Hole@{k_value}"] = len(hole_docs) / k_value

        return hole

class TopKAccuracyFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> Dict[str, float]:
        top_k_acc = {}
        top_k_acc[f"Accuracy@{k_value}"] = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])
        for relevant_doc_id in query_relevant_docs:
            if relevant_doc_id in top_hits:
                top_k_acc[f"Accuracy@{k_value}"] += 1.0
                break

        return top_k_acc