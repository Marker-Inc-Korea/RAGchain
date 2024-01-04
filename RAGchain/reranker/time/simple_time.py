from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage


class SimpleTimeReranker(BaseReranker):
    """Rerank passages by their content_datetime only. It is simple reranker for time-aware RAG."""

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        passages = input.passages
        sorted_passages = sorted(passages, key=lambda p: p.content_datetime, reverse=True)
        input.passages = sorted_passages
        return input

    def rerank(self, passages: List[Passage]) -> List[Passage]:
        """
        Rerank passages by their content_datetime only.
        :param passages: list of passages to be reranked.
        """
        return sorted(passages, key=lambda p: p.content_datetime, reverse=True)
