from abc import ABC, abstractmethod
from typing import List

from KoPrivateGPT.schema import Passage


class SlimVectorStore(ABC):
    def embed_passage(self, passage: Passage):
        """
        Embed a passage
        """
        self.embed_passages([passage])

    @abstractmethod
    def embed_passages(self, passages: List[Passage]):
        """
        Embed multiple passages
        """
        pass
