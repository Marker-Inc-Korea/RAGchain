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
        """
        pass
