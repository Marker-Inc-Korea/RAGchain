from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Type

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.schema import Passage, RetrievalResult


class BaseWebSearch(Runnable[Union[str, Tuple[str, int]], RetrievalResult], ABC):
    """
    Abstract class for using a web search engine for passage contents.
    """

    @abstractmethod
    def get_search_data(self, query: str, num_results: int = 5, ) -> List[Passage]:
        """
        Abstract method for searching passages from the web search engine.
        """
        pass

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        if isinstance(input, str):
            passages = self.get_search_data(input)
            return RetrievalResult(query=input, passages=passages, scores=self.__make_scores(len(passages)))
        elif isinstance(input, Tuple):
            passages = self.get_search_data(input[0], num_results=input[1])
            return RetrievalResult(query=input[0], passages=passages, scores=self.__make_scores(len(passages)))
        else:
            raise ValueError(f"Input type must be str or Tuple[str, int], but got {type(input)}")

    @property
    def InputType(self) -> Type[Input]:
        return Union[str, Tuple[str, int]]

    @property
    def OutputType(self) -> Type[Output]:
        return List[Passage]

    @staticmethod
    def __make_scores(retrieved_length: int):
        # Make scores with range of 1 to 0. The length must be retrieve_length.
        return [i / retrieved_length for i in range(retrieved_length, 0, -1)]
