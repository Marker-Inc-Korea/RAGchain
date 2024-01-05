from abc import ABC, abstractmethod
from typing import List, Optional, Type

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.schema import Passage, RetrievalResult


class BaseCompressor(Runnable[RetrievalResult, RetrievalResult], ABC):
    @abstractmethod
    def compress(self, passages: List[Passage], **kwargs) -> List[Passage]:
        pass

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """
        Compress the passages in input and return the compressed passages.
        It gets compression algorithm's parameters from config.
        Key name is 'compressor_params'.
        Set parameters at configurable to dict.
        Example:
            runnable.invoke(retrieval_result, config={"configurable": {"compressor_params": {"n_clusters": 3}}})

        Important!
        The scores of the passages will be removed.
        It is recommended to use this module after all retrievals and reranking passages,
        before you put the passages into LLM.
        """

        params = config['configurable'].get('compressor_params', {}) if config is not None else {}
        compressed_passages = self.compress(input.passages, **params)
        return RetrievalResult(
            query=input.query,
            passages=compressed_passages,
            scores=[],
        )

    @property
    def InputType(self) -> Type[Input]:
        return RetrievalResult

    @property
    def OutputType(self) -> Type[Output]:
        return RetrievalResult
