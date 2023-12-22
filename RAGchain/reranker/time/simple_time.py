from typing import List

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class SimpleTimeReranker(BaseReranker):

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        pass
