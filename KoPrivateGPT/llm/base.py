from abc import ABC, abstractmethod
from typing import List
from KoPrivateGPT.schema import Passage


class BaseLLM(ABC):
    @abstractmethod
    def ask(self, query: str, passages: List[Passage]) -> str:
        """
        Ask a question to the LLM model with passages and get answer and used passages
        """
        pass
