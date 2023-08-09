from abc import ABC, abstractmethod
from typing import List

from KoPrivateGPT.schema import Passage


class BaseRetrieval(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        pass

    @abstractmethod
    def ingest(self, passages: List[Passage]):
        pass
