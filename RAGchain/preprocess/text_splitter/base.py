from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document

from RAGchain.schema import Passage


class BaseTextSplitter(ABC):
    """
    Base class for text splitters.
    At this framework, we use our own text splitter instead of the one from langchain.
    """
    def split_documents(self, documents: List[Document]) -> List[List[Passage]]:
        """
        Split a list of documents into passages.
        The return passages will be 2d list, where the first dimension is the document index,
        and the second dimension is the passage index.
        """
        return [self.split_document(document) for document in documents]

    @abstractmethod
    def split_document(self, document: Document) -> List[Passage]:
        """
        Split a document into passages.
        """
        pass
