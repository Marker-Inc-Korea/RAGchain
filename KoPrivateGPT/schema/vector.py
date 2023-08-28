from typing import List, Union
from uuid import UUID

from langchain.load.serializable import Serializable


class Vector(Serializable):
    """Class for storing a vector for VectorDB"""
    vector: List[float]
    passage_id: Union[UUID, str]
