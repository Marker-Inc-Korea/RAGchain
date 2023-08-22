from typing import List

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.reranker.base import BaseReranker

from pygaggle.rerank.transformer import MonoT5


class MonoT5(BaseReranker):

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        pass

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("MonoT5 does not support sliding window reranking")