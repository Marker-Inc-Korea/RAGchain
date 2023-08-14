from typing import Optional, Union

from langchain.load.serializable import Serializable
from langchain.schema import Document
from pydantic import Field
from uuid import UUID, uuid4


class Passage(Serializable):
    """Class for storing a passage and metadatas"""
    id: Union[UUID, str] = Field(default_factory=uuid4)
    content: str
    filepath: str
    previous_passage_id: Optional[UUID]
    next_passage_id: Optional[UUID]
    metadata_etc: dict = Field(default_factory=dict)

    def to_document(self) -> Document:
        metadata = self.metadata_etc.copy()
        metadata['id'] = self.id
        metadata['content'] = self.content
        metadata['filepath'] = self.filepath
        metadata['previous_passage_id'] = self.previous_passage_id
        metadata['next_passage_id'] = self.next_passage_id
        return Document(page_content=self.content, metadata=metadata)

    @classmethod
    def from_dict(cls, data: dict):
        cls.id = data["_id"]
        cls.content = data["content"]
        cls.filepath = data["filepath"]
        cls.previous_passage_id = data["previous_passage_id"]
        cls.next_passage_id = data["next_passage_id"]
        cls.metadata_etc = data["metadata_etc"]
        return cls(**data)
