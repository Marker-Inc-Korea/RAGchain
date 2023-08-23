from typing import List

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.reranker.base import BaseReranker


class MonoT5Reranker(BaseReranker):

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        pass

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("MonoT5 does not support sliding window reranking")
