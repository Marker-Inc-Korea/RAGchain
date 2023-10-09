from abc import ABC, abstractmethod
from typing import List

from KoPrivateGPT.schema import Passage


class BaseReranker(ABC):
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

    @abstractmethod
    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        """
        Reranks a list of passages based on a specific ranking algorithm with sliding window.
        This function is useful when the model input token size is limited like LLMs.

        :param passages: (List[Passage]): The list of passages to be reranked.
        :param query: str: The query that was used for retrieving the passages.
        :param window_size: (int): The size of the sliding window used for reranking.

        :return: List[Passage]: The reranked list of passages.

        """
        pass
