from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
import pytrec_eval
import json
import math
from operator import itemgetter


class BaseRetrievalEvaluationFactory(ABC):

    def eval(self, solution: Dict[str, int],
             pred: Dict[str, float],
             k: int) -> float:
        assert k > 0, "k must be greater than 0"
        metric = self.retrieval_metric_function(solution, pred, k)

        return metric

    @abstractmethod
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        pass


class APFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        ap = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        count_relevant = 0

        for index, doc_id in enumerate(top_hits):
            if doc_id in query_relevant_docs:
                count_relevant += 1
                precision_at_relevant_doc = count_relevant / (index + 1)
                ap += precision_at_relevant_doc

        return ap


class NDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        ndcg = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg = sum((2 ** relevance - 1) / math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.values()))
        dcg = sum((2 ** top_hits_ideal[doc_id] - 1) / math.log2(i + 2) if doc_id in solution else 0
                  for i, doc_id in enumerate(top_hits))

        ndcg += dcg / idcg if idcg > 0 else 0  # ndcg need to deal whole query, so this is not complete ndcg

        return ndcg


class CGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        cg = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        cg += sum(top_hits_ideal[doc_id] if doc_id in solution else 0 for doc_id in top_hits)

        return cg


class IndDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        dcg_ind = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        dcg_ind += sum((2 ** top_hits_ideal[doc_id] - 1) / math.log2(i + 2) if doc_id in solution else 0
                       for i, doc_id in enumerate(top_hits))

        return dcg_ind


class DCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        dcg = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        dcg += sum(top_hits_ideal[doc_id] / math.log2(i + 2) if doc_id in solution else 0
                   for i, doc_id in enumerate(top_hits))

        return dcg


class IndIDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        idcg_ind = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg_ind += sum(
            (2 ** relevance - 1) / math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.values()))

        return idcg_ind


class IDCGFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        idcg = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg += sum(relevance / math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.values()))

        return idcg


class RecallFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        recall = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        assert len(query_relevant_docs) > 0, "pred must have at least one document"
        recall += len(relevant_retrieved_docs) / len(query_relevant_docs)

        return recall


class PrecisionFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        precision = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        precision += len(relevant_retrieved_docs) / len(top_hits) if len(top_hits) > 0 else 0

        return precision


class RRFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        """
         Reciprocal Rank (RR) is the reciprocal of the rank of the first relevant item.
         Mean of RR in whole querys is MRR.
        """
        rr = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        for rank, doc_id in enumerate(top_hits):
            if doc_id in query_relevant_docs:
                rr += 1.0 / (rank + 1)
                break

        return rr


class HoleFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        hole = 0.0

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        hole_docs = [doc_id for doc_id in top_hits if doc_id not in query_relevant_docs]
        hole += len(hole_docs) / len(top_hits)

        return hole


class TopKAccuracyFactory(BaseRetrievalEvaluationFactory):
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        top_k_acc = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])
        for relevant_doc_id in query_relevant_docs:
            if relevant_doc_id in top_hits:
                top_k_acc += 1.0
                break

        return top_k_acc
