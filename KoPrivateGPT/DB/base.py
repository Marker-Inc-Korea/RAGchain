from abc import ABC, abstractmethod
from typing import List, Union, Any
from KoPrivateGPT.schema import Passage
from uuid import UUID

from KoPrivateGPT.schema.db_path import DBOrigin


class BaseDB(ABC):
    @property
    @abstractmethod
    def db_type(self) -> str:
        pass

    @abstractmethod
    def create(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_or_load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, passages: List[Passage]):
        pass

    @abstractmethod
    def fetch(self, ids: List[UUID]) -> List[Passage]:
        pass

    @abstractmethod
    def search(self, filter: Any) -> List[Passage]:
        pass

    @abstractmethod
    def get_db_origin(self) -> DBOrigin:
        pass
