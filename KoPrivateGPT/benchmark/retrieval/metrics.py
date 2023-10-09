import math
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict
import warnings

class BaseRetrievalMetric(ABC):
    def __init__(self):
        self._metric_name = None

    @property
    def metric_name(self):
        return str(self._metric_name)

    def eval(self, solution: Dict[str, int],
             pred: Dict[str, float],
             k: int) -> float:
        assert k > 0, "k must be greater than 0"
        assert len(pred) >= k, "k must be less than or equal to the number of predictions"

        metric = self.retrieval_metric_function(solution, pred, k)

        return metric

    @abstractmethod
    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:

        pass


class AP(BaseRetrievalMetric):
    def __init__(self):
        super().__init__()
        self._metric_name = "AP"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        temp_sum = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        count_relevant = 0

        for index, doc_id in enumerate(top_hits):
            if doc_id in query_relevant_docs:
                count_relevant += 1
                precision_at_relevant_doc = count_relevant / (index + 1)
                temp_sum += precision_at_relevant_doc

        ap = temp_sum / count_relevant if count_relevant > 0 else 0.0

        return ap


class NDCG(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "NDCG"

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


class CG(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "CG"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        cg = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        cg += sum(top_hits_ideal[doc_id] if doc_id in solution else 0 for doc_id in top_hits)

        return cg


class IndDCG(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "Ind_DCG"

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


class DCG(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "DCG"

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


class IndIDCG(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "Ind_IDCG"

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


class IDCG(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "IDCG"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        idcg = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        top_hits_ideal = dict(
            sorted({doc_id: solution.get(doc_id, 0) for doc_id in top_hits}.items(), key=itemgetter(1), reverse=True))

        idcg += sum(relevance / math.log2(i + 2) for i, relevance in enumerate(top_hits_ideal.values()))

        return idcg


class Recall(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "Recall"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        recall = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        recall += len(relevant_retrieved_docs) / len(query_relevant_docs)

        return recall


class Precision(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "Precision"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        precision = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        precision += len(relevant_retrieved_docs) / len(top_hits) if len(top_hits) > 0 else 0

        return precision


class RR(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "RR"

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


class Hole(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "Hole"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        hole = 0.0

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        hole_docs = [doc_id for doc_id in top_hits if doc_id not in query_relevant_docs]
        hole += len(hole_docs) / len(top_hits)

        return hole


class TopKAccuracy(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "TopK_Accuracy"

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


class ExactlyMatch(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "EM"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        EM = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])
        if set(solution.keys()) == set(pred.keys()):
            EM += 1.0

        return EM


class F1(BaseRetrievalMetric):
    def __init__(self):
        self._metric_name = "F1_score"

    def retrieval_metric_function(self, solution: Dict[str, int],
                                  pred: Dict[str, float],
                                  k_value: int = 1) -> float:
        recall = 0.0
        precision = 0.0
        f1 = 0.0

        top_hits = [item[0] for item in sorted(pred.items(), key=lambda item: item[1], reverse=True)[:k_value]]

        query_relevant_docs = set([doc_id for doc_id in solution if solution[doc_id] > 0])

        relevant_retrieved_docs = [doc_id for doc_id in top_hits if doc_id in query_relevant_docs]

        recall += len(relevant_retrieved_docs) / len(query_relevant_docs)

        precision += len(relevant_retrieved_docs) / len(top_hits) if len(top_hits) > 0 else 0

        f1 += 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1
