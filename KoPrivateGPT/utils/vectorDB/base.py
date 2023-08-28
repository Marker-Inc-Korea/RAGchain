from abc import ABC, abstractmethod
from typing import List, Union
from uuid import UUID

from KoPrivateGPT.schema.vector import Vector


class BaseVectorDB(ABC):
    @abstractmethod
    def add_vectors(self, vectors: List[Vector]):
        pass

    @abstractmethod
    def similarity_search(self, query_vectors: List[float], top_k: int = 5) -> tuple[
        List[Union[UUID, str]], List[float]]:
        """
            Return top_k passage_ids and similarity scores.
            At Default, you must return each similarity scores.
        """
        pass

    @abstractmethod
    def delete_all(self):
        pass

    @abstractmethod
    def get_db_type(self) -> str:
        pass

    def _str_to_uuid(self, input: str) -> Union[str, UUID]:
        try:
            return UUID(input)
        except:
            return input
