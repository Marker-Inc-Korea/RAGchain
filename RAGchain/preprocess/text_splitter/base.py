from abc import ABC, abstractmethod
from datetime import datetime
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
            filepath = metadata_etc.pop('source', None)
            if filepath is None:
                raise ValueError(f"source must be provided in metadata, but got {metadata_etc}")

            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(split_documents) - 1 else None
            passage = Passage(id=uuid,
                              content=split_document.page_content,
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            # put content_datetime
            content_datetime = metadata_etc.pop('content_datetime', None)
            if content_datetime is not None:
                if isinstance(content_datetime, str):
                    content_datetime = datetime.strptime(content_datetime, '%Y-%m-%d %H:%M:%S')
                if not isinstance(content_datetime, datetime):
                    raise TypeError(f"content_datetime must be datetime, but got {type(content_datetime)}")
                passage.content_datetime = content_datetime

            # put importance
            importance = metadata_etc.pop('importance', None)
            if importance is not None:
                if not isinstance(importance, int):
                    raise TypeError(f"importance must be int, but got {type(importance)}")
                passage.importance = importance

            passages.append(passage)
        print(f"Split into {len(passages)} passages")
        return passages
