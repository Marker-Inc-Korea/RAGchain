from abc import ABC, abstractmethod
from typing import List

from RAGchain.schema import Passage


class SlimVectorStore(ABC):
    """
    A vector store stores only passage_id and vector.
    However, default VectorStore from langchian stores all metadata and contents, so its size is huge.
    Using SlimVectorStore, you can reduce the size of vector store.
    """
    def add_passage(self, passage: Passage):
        """
        Embed a passage
        """
        self.add_passages([passage])

    @abstractmethod
    def add_passages(self, passages: List[Passage]):
        """
        Embed multiple passages
        """
        pass
