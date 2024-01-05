from datetime import datetime
from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage, RetrievalResult


class WeightedTimeReranker(BaseReranker):
    """
    Rerank passages by their content_datetime and relevance score.
    First, relevance score must be normalized to [0, 1] range.
    And calculate the combined score by the following formula:
        score = (1 - decay_rate) ** hours_passed + relevance_score

    The larger the decay_rate, the score from the past will be lowed.
    """

    def __init__(self, decay_rate: float = 0.01):
        """
        :param decay_rate: decay rate of time weight. The smaller the value, the more important the time weight.
        """
        super().__init__()
        self.decay_rate = decay_rate

    def rerank(self, passages: List[Passage], scores: List[float]) -> List[Passage]:
        """
        :param passages: list of passages to be reranked.
        :param scores: list of relevance scores of passages.
        """
        retrieval_result = RetrievalResult(query="", passages=passages, scores=scores)
        return self.invoke(retrieval_result).passages

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        now = datetime.now()
        scores = input.scores
        passages = input.passages
        # normalize scores
        scaled_scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
        combined_scores = [self.__get_combined_score(passage, score=score, now=now)
                           for passage, score in zip(passages, scaled_scores)]
        sorted_passages, sorted_scores = zip(*sorted(zip(passages, combined_scores), key=lambda x: x[1], reverse=True))
        input.passages = list(sorted_passages)
        input.scores = list(sorted_scores)
        return input

    def __get_combined_score(self, passage: Passage, score: float, now: datetime = datetime.now()):
        passed_hours = (now - passage.content_datetime).total_seconds() / 3600
        return ((1.0 - self.decay_rate) ** passed_hours) + score
