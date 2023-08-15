from uuid import UUID

from langchain.load.serializable import Serializable
from typing import List


class Vector(Serializable):
    """Class for storing a vector for VectorDB"""
    vector: List[float]
    passage_id: UUID
