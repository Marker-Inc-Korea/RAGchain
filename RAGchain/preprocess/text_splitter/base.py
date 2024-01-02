import itertools
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from langchain.schema import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

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

    def as_runnable(self):
        """
        Return a runnable version of this text splitter.
        """
        return RunnableTextSplitter(self)


class RunnableTextSplitter(Runnable[List[Document], List[Passage]]):
    def __init__(self, text_splitter: BaseTextSplitter):
        self.text_splitter = text_splitter

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return itertools.chain.from_iterable(self.text_splitter.split_documents(input))

    @property
    def InputType(self) -> Type[Input]:
        return List[Document]

    @property
    def OutputType(self) -> Type[Output]:
        return List[Passage]
