from abc import ABC, abstractmethod
from typing import List, Type

from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input

from RAGchain.schema import Passage, RetrievalResult


class BaseReranker(Runnable[RetrievalResult, RetrievalResult], ABC):
    @abstractmethod
    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        """
        Reranks a list of passages based on a specific ranking algorithm.

        :param passages: A list of Passage objects representing the passages to be reranked.
        :type passages: List[Passage]
        :param query: str: The query that was used for retrieving the passages.
        :return: The reranked list of passages.
        :rtype: List[Passage]

        """
        pass

    @property
    def InputType(self) -> Type[Input]:
        return RetrievalResult

    @property
    def OutputType(self) -> Type[RetrievalResult]:
        return RetrievalResult
