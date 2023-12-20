from datetime import datetime
from typing import Optional, Union, List
from uuid import UUID, uuid4

from langchain.load.serializable import Serializable
from langchain.schema import Document
from pydantic import Field


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
    previous_passage_id: Optional[Union[UUID, str]]
    """Previous passage's id. If this is the first passage, this value should be None."""
    next_passage_id: Optional[Union[UUID, str]]
    """Next passage's id. If this is the last passage, this value should be None."""
    metadata_etc: dict = Field(default_factory=dict)
    """Arbitrary metadata about the passage."""

    def to_document(self) -> Document:
        metadata = self.metadata_etc.copy()
        metadata['id'] = self.id
        metadata['content'] = self.content
        metadata['filepath'] = self.filepath
        metadata['content_datetime'] = self.content_datetime
        metadata['previous_passage_id'] = self.previous_passage_id
        metadata['next_passage_id'] = self.next_passage_id
        return Document(page_content=self.content, metadata=metadata)

    def to_dict(self):
        return {
            "_id": self.id,
            "content": self.content,
            "filepath": self.filepath,
            "content_datetime": self.content_datetime,
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
            self.previous_passage_id == other.previous_passage_id and \
            self.next_passage_id == other.next_passage_id and \
            self.metadata_etc == other.metadata_etc

    @staticmethod
    def make_prompts(passages: List['Passage']) -> str:
        return "\n".join([passage.content for passage in passages])
