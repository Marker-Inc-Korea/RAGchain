from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union, Optional
from uuid import UUID

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from RAGchain.schema import Passage
from RAGchain.schema.db_origin import DBOrigin


class BaseDB(Runnable[List[Passage], List[Passage]], ABC):
    """
    Abstract class for using a database for passage contents.
    """
    @property
    @abstractmethod
    def db_type(self) -> str:
        """
        The type of the database. This attribute should be implemented by the subclasses.
        """
        pass

    @abstractmethod
    def create(self, *args, **kwargs):
        """
        Abstract method for creating a new database.
        """
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        """Abstract method for loading existed database."""
        pass

    @abstractmethod
    def create_or_load(self, *args, **kwargs):
        """Abstract method for creating a new database or loading existing database."""
        pass

    @abstractmethod
    def save(self, passages: List[Passage]):
        """Abstract method for saving passages to the database."""
        pass

    @abstractmethod
    def fetch(self, ids: List[UUID]) -> List[Passage]:
        """Abstract method for fetching passages from the database based on their passage IDs."""
        pass

    @abstractmethod
    def search(self,
               id: Optional[List[Union[UUID, str]]] = None,
               content: Optional[List[str]] = None,
               filepath: Optional[List[str]] = None,
               content_datetime_range: Optional[List[tuple[datetime, datetime]]] = None,
               importance: Optional[List[int]] = None,
               **kwargs
               ) -> List[Passage]:
        """
            search Passage from DB using filter Dict.
            This function can search Passage using 'id', 'content', 'filepath' and 'metadata_etc'.
            You can add more search condition in metadata_etc using **kwargs.
            This function is an implicit AND operation.

            :param id: (Optional[List[Union[UUID, str]]]): List of Passage ID to search.
            :param content: (Optional[List[str]]): List of Passage content to search.
            :param filepath: (Optional[List[str]]): List of Passage filepath to search.
            :param content_datetime_range: (Optional[List[tuple[datetime.datetime, datetime.datetime]]]): List of content_datetime range to search.
            You can set start_time at the first element in tuple and end_time at the second element in tuple.
            Multiple time range search is possible, and the search condition is OR operation in time range.
            :param importance: (Optional[List[int]]): List of passages' importance to search.
            **kwargs: Additional metadata to search.
        """
        pass

    @abstractmethod
    def get_db_origin(self) -> DBOrigin:
        """DBOrigin: Abstract method for retrieving DBOrigin of the database."""
        pass

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        self.create_or_load()
        self.save(input)
        return input

    @property
    def InputType(self) -> type[Input]:
        return List[Passage]

    @property
    def OutputType(self) -> type[Output]:
        return List[Passage]
