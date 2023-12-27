from abc import ABC, abstractmethod
from typing import List

from RAGchain.schema import Passage


class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, passages: List[Passage]) -> List[Passage]:
        pass
