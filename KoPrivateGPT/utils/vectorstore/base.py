from abc import ABC, abstractmethod
from typing import List

from KoPrivateGPT.schema import Passage


class SlimVectorStore(ABC):
    def add_passage(self, passage: Passage):
        """
        Embed a passage
        """
        self.add_passages([passage])

    @abstractmethod
    def add_passages(self, passages: List[Passage]):
        """
        Embed multiple passages
        Must include "passage_id" at metadatas.
        The value of "passage_id" must be string, because many vector stores don't support UUID.
        Must include empty string at document contents.
        """
        pass
