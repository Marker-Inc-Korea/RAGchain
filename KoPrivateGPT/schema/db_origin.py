import json

from langchain.load.serializable import Serializable
from pydantic import Field


class DBOrigin(Serializable):
    """Class for storing a db_type and db_path: dict"""
    db_type: str
    db_path: dict = Field(default_factory=dict)

    def to_json(self) -> json:
        return {
            "db_type": self.db_type,
            "db_path": self.db_path
        }
