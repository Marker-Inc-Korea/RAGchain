from abc import ABC, abstractmethod
from typing import List, Union, Optional
from uuid import UUID

from KoPrivateGPT.schema import Passage
from KoPrivateGPT.schema.db_origin import DBOrigin


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
    def search(self,
               id: Optional[List[Union[UUID, str]]] = None,
               content: Optional[List[str]] = None,
               filepath: Optional[List[str]] = None,
               **kwargs
               ) -> List[Passage]:
        """
            Search Passage from DB using filter Dict.
            This function can search Passage using 'id', 'content', 'filepath' and 'metadata_etc'.
            You can add more search condition in metadata_etc using **kwargs.
            This function is an implicit AND operation.
        """
        pass

    @abstractmethod
    def get_db_origin(self) -> DBOrigin:
        pass
