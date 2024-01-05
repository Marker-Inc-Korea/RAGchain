from abc import ABC, abstractmethod
from typing import Optional, List, Type

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.schema import Passage, RetrievalResult


class BaseWebSearch(Runnable[str, RetrievalResult], ABC):
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
        """
        Invoke the WebSearch module.
        :param input: A query string.
        :param config: You can set num_results in config.
        The configurable key is "web_search_options".
        For example,
            runnable.invoke("your search query", config={"configurable": {"web_search_options": {"num_results": 10}}})
        """
        retrieval_option = config['configurable'].get('web_search_options', {}) if config is not None else {}
        passages = self.get_search_data(input, **retrieval_option)
        return RetrievalResult(query=input, passages=passages, scores=self.__make_scores(len(passages)))

    @property
    def InputType(self) -> Type[Input]:
        return str

    @property
    def OutputType(self) -> Type[Output]:
        return RetrievalResult

    @staticmethod
    def __make_scores(retrieved_length: int):
        # Make scores with range of 1 to 0. The length must be retrieve_length.
        return [i / retrieved_length for i in range(retrieved_length, 0, -1)]
