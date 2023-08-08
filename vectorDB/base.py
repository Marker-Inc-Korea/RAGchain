from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple
from langchain.schema import Document


class BaseVectorDB(ABC):
    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass

    @abstractmethod
    def add_documents(self, docs: List[Document]):
        pass

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
            Return top_k documents
            At Default, you must return each similarity scores.
        """
        pass

    @abstractmethod
    def delete_all(self):
        pass

    @abstractmethod
    def get_db_type(self) -> str:
        pass
