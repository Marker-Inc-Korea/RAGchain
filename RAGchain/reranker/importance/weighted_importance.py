from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage, RetrievalResult


class WeightedImportanceReranker(BaseReranker):
    """
    Rerank passages by their importance and relevance score.
    First, relevance score and importance must be normalized to [0, 1] range.
    And calculate the combined score by the following formula:
        score = (weight) * importance + (1 - weight) * relevance_score

    The larger the weight, the more important the importance is.
    """

    def __init__(self, importance_weight: float = 0.5):
        """
        :param importance_weight: weight of importance. The larger the value, the more important the importance is.
        """
        super().__init__()
        self.importance_weight = importance_weight

    def rerank(self, passages: List[Passage], scores: List[float]) -> List[Passage]:
        """
        :param passages: list of passages to be reranked.
        :param scores: list of relevance scores of passages.
        """
        result = self.invoke(RetrievalResult(query='', passages=passages, scores=scores))
        return result.passages

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        scores = input.scores
        passages = input.passages
        normalize_rel_scores = self.__normalize(scores)
        normalize_importance = self.__normalize([passage.importance for passage in passages])
        combined_scores = [self.__get_combined_score(rel_score, importance) for rel_score, importance in
                           zip(normalize_rel_scores, normalize_importance)]
        sorted_passages, sorted_scores = zip(*sorted(zip(passages, combined_scores), key=lambda x: x[1], reverse=True))
        input.passages = list(sorted_passages)
        input.scores = list(sorted_scores)
        return input

    @staticmethod
    def __normalize(scores: List[float]) -> List[float]:
        """
        :param scores: list of scores to be normalized.
        """
        return [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]

    def __get_combined_score(self, rel_score: float, importance: float):
        return self.importance_weight * importance + (1 - self.importance_weight) * rel_score
