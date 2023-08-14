from abc import ABC, abstractmethod
from typing import List, Union
from uuid import UUID

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage


class BaseRetrieval(ABC):
    @abstractmethod
    def retrieve(self, query: str, db: BaseDB, top_k: int = 5, *args, **kwargs) -> List[Passage]:
        pass

    @abstractmethod
    def ingest(self, passages: List[Passage]):
        pass

    @abstractmethod
    def retrieve_id(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Union[str, UUID]]:
        pass
