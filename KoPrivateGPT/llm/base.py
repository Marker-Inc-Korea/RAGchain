from abc import ABC, abstractmethod
from typing import List

from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class BaseLLM(ABC):
    @abstractmethod
    def ask(self, query: str) -> tuple[str, List[Passage]]:
        """
        Ask a question to the LLM model and get answer and used passages
        """
        pass
