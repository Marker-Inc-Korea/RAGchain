from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from KoPrivateGPT.schema import Passage


class BaseTextSplitter(ABC):
    def split_documents(self, documents: List[Document]) -> List[List[Passage]]:
        return [self.split_document(document) for document in documents]

    @abstractmethod
    def split_document(self, document: Document) -> List[Passage]:
        pass
