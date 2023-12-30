from datetime import datetime
from typing import Optional, Union, List, Dict
from uuid import UUID, uuid4

from langchain.load.serializable import Serializable
from langchain.schema import Document
from pydantic import Field, Extra


class Passage(Serializable):
    """Class for storing a passage and metadatas"""

    id: Union[UUID, str] = Field(default_factory=uuid4)
    """Unique identifier for the passage. You can use string or UUID. By default, create new UUID for new passage."""
    content: str
    """String text."""
    filepath: str
    """Filepath of the passage."""
    content_datetime: datetime = Field(default_factory=datetime.now)
    """Datetime when the passage content is created or edited. Everytime passge content changes, this value should be 
    updated."""
    importance: int = Field(default=0)
    """Importance of the passage. The higher the value, the more important the passage is. It can be minus value. 
    The default is 0."""
    previous_passage_id: Optional[Union[UUID, str]]
    """Previous passage's id. If this is the first passage, this value should be None."""
    next_passage_id: Optional[Union[UUID, str]]
    """Next passage's id. If this is the last passage, this value should be None."""
    metadata_etc: dict = Field(default_factory=dict)
    """Arbitrary metadata about the passage."""

    # forbid to use another parameter
    class Config:
        extra = Extra.forbid

    def to_document(self) -> Document:
        metadata = self.metadata_etc.copy()
        metadata['id'] = self.id
        metadata['content'] = self.content
        metadata['filepath'] = self.filepath
        metadata['content_datetime'] = self.content_datetime
        metadata['importance'] = self.importance
        metadata['previous_passage_id'] = self.previous_passage_id
        metadata['next_passage_id'] = self.next_passage_id
        return Document(page_content=self.content, metadata=metadata)

    def to_dict(self):
        return {
            "_id": self.id,
            "content": self.content,
            "filepath": self.filepath,
            "content_datetime": self.content_datetime,
            "importance": self.importance,
            "previous_passage_id": self.previous_passage_id,
            "next_passage_id": self.next_passage_id,
            "metadata_etc": self.metadata_etc
        }

    def __eq__(self, other):
        if isinstance(other, Passage):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def is_exactly_same(self, other):
        return self.id == other.id and \
            self.content == other.content and \
            self.filepath == other.filepath and \
            self.content_datetime == other.content_datetime and \
            self.importance == other.importance and \
            self.previous_passage_id == other.previous_passage_id and \
            self.next_passage_id == other.next_passage_id and \
            self.metadata_etc == other.metadata_etc

    @staticmethod
    def make_prompts(passages: List['Passage']) -> str:
        return "\n".join([passage.content for passage in passages])

    def copy(self, *args, **kwargs):
        self_params = self.dict()
        for key in list(kwargs.keys()):
            self_params.pop(key)
        return Passage(**self_params, **kwargs)

    def reset_id(self):
        self.id = uuid4()
        return self

    @classmethod
    def from_documents(cls, documents: List[Document]) -> List['Passage']:
        """
        Convert a list of documents to a list of passages.
        metadata with 'source' key is required. It will convert to filepath filed.
        metadat with 'content_datetime' key is optional. It will convert to content_datetime field.
        It can be datetime.datetime object, or string with '%Y-%m-%d %H:%M:%S' format.
        metadata with 'importance' key is optional. It will convert to importance field. It must be int.
        :param documents: A list of documents.
        """
        passages = []
        ids = [uuid4() for _ in range(len(documents))]
        for i, (split_document, uuid) in enumerate(zip(documents, ids)):
            metadata_etc = split_document.metadata.copy()
            filepath = metadata_etc.pop('source', None)
            if filepath is None:
                raise ValueError(f"source must be provided in metadata, but got {metadata_etc}")

            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(ids) - 1 else None
            passage = cls(id=uuid,
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

    @classmethod
    def from_search(cls, search_results: List[Dict[str, str]]) -> List['Passage']:
        """
        Convert a list of search results to a list of passages.
        :param search_results: A list of search results.
        """
        if len(search_results) == 0:
            return []
        passages = []
        ids = [uuid4() for _ in range(len(search_results))]
        for i, (search_results, uuid) in enumerate(zip(search_results, ids)):
            metadata_etc = {"title": search_results["title"]}
            filepath = search_results["link"]
            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(ids) - 1 else None
            passage = cls(id=uuid,
                          content=search_results["snippet"],
                          filepath=filepath,
                          previous_passage_id=previous_passage_id,
                          next_passage_id=next_passage_id,
                          metadata_etc=metadata_etc)
            passages.append(passage)
        return passages
