from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class SimpleImportanceReranker(BaseReranker):
    """Rerank passages by their importance only. It is simple reranker for importance-aware RAG"""

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        sorted_pairs = sorted(zip(input.passages, input.scores), key=lambda p: p[0].importance, reverse=True)
        sorted_passages, sorted_scores = zip(*sorted_pairs)
        input.passages = list(sorted_passages)
        input.scores = list(sorted_scores)
        return input

    def rerank(self, passages: List[Passage]) -> List[Passage]:
        """
        Rerank passages by their content_datetime only.
        :param passages: list of passages to be reranked.
        """
        return sorted(passages, key=lambda p: p.importance, reverse=True)
