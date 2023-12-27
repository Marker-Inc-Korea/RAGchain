from typing import List

from RAGchain.schema import Passage


class SimpleTimeReranker:
    """Rerank passages by their content_datetime only. It is simple reranker for time-aware RAG."""

    def rerank(self, passages: List[Passage]) -> List[Passage]:
        """
        Rerank passages by their content_datetime only.
        :param passages: list of passages to be reranked.
        """
        return sorted(passages, key=lambda p: p.content_datetime, reverse=True)
