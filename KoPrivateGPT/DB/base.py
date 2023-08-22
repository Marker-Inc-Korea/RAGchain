from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict
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
    def search(self, filter_dict: Dict[str, Any]) -> List[Passage]:
        """
            Search Passage from DB using filter Dict.
            This function can search Passage using 'content', 'filepath' and 'metadata_etc' key.
            :param filter_dict: Dict[str, str]
            The key of filter_dict must be key that you want to search.
            Search function search Passage that matches value.
        """
        pass

    @abstractmethod
    def get_db_origin(self) -> DBOrigin:
        pass
