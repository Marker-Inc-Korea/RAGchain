from abc import ABC, abstractmethod
from typing import List
from uuid import uuid4

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

    def docs_to_passages(self, split_documents: List[Document]) -> List[Passage]:
        passages = []
        ids = [uuid4() for _ in range(len(split_documents))]
        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            metadata_etc = split_document.metadata.copy()
            filepath = metadata_etc.pop('source')
            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(split_documents) - 1 else None
            passage = Passage(id=uuid,
                              content=split_document.page_content,
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            passages.append(passage)
        print(f"Split into {len(passages)} passages")
        return passages
