from typing import List

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class BM25Reranker(BaseReranker):
    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        pass

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("BM25Reranker doesn't support sliding window reranking.")
