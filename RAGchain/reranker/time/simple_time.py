from typing import List

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class SimpleTimeReranker(BaseReranker):
    """Rerank passages by their content_datetime only. It is simple reranker for time-aware RAG."""
    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        return sorted(passages, key=lambda p: p.content_datetime, reverse=True)
