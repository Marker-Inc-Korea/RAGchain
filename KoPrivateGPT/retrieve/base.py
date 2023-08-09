from abc import ABC, abstractmethod
from typing import List, Dict

from langchain.schema import Document


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, *args, **kwargs) -> List[Document]:
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, documents: List[Document], *args, **kwargs):
        pass

    @abstractmethod
    def save_one(self, document: Document, *args, **kwargs):
        pass

    @abstractmethod
    def delete(self, ids: List[str], *args, **kwargs):
        pass

    @abstractmethod
    def delete_one(self, id: str, *args, **kwargs):
        pass

    @abstractmethod
    def delete_all(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, documents: List[Document], *args, **kwargs):
        """
            Update documents with ids
            inside each document, id must be set.
        """
        pass

    @abstractmethod
    def update_one(self, document: Document, *args, **kwargs):
        """
            Update documents with ids
            inside each document, id must be set.
        """
        pass
