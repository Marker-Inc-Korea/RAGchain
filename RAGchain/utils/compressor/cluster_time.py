from typing import List, Iterator

from RAGchain.schema import Passage
from RAGchain.utils.compressor.base import BaseCompressor
from RAGchain.utils.semantic_clustering import SemanticClustering


class ClusterTimeCompressor(BaseCompressor):
    """
    Compress passages by semantically clustering them and then selecting the most recent passage from each cluster.
    """

    def __init__(self, semantic_cluster: SemanticClustering,
                 split_by_sentences: bool = False):
        """
        :param semantic_cluster: SemanticClustering object used to cluster the passages.
        :param split_by_sentences: Whether to split the passages into sentences before clustering.
        It can be helpful that each passage size is big or contain whole different meanings in one passage.
        Default is False.
        """
        self.semantic_cluster = semantic_cluster
        self.split_by_sentences = split_by_sentences

    def compress(self, passages: List[Passage], **kwargs) -> List[Passage]:
        """
        :param passages: list of passages to be compressed.
        :param kwargs: kwargs for clustering algorithm.
        """
        pass

    @staticmethod
    def _split_sentences(passages: List[Passage]) -> Iterator[Passage]:
        """
        split passages into sentences. Keep the original passage id and other params. Only change the content.
        So, the returned passages have same id and other params, but different content.
        """
        for passage in passages:
            sentences = passage.content.split(".")
            for sentence in sentences:
                yield passage.copy(content=sentence.strip())
