from typing import List

from RAGchain.schema import Passage


class SimpleImportanceReranker:
    """Rerank passages by their importance only. It is simple reranker for importance-aware RAG"""

    def rerank(self, passages: List[Passage]) -> List[Passage]:
        """
        Rerank passages by their importance only.
        :param passages: list of passages to be reranked.
        """
        return sorted(passages, key=lambda p: p.importance, reverse=True)
