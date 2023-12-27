import pytest

from RAGchain.schema import Passage
from RAGchain.utils.compressor.cluster_time import ClusterTimeCompressor
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.semantic_clustering import SemanticClustering


@pytest.fixture
def cluster_time_compressor():
    compressor = ClusterTimeCompressor(semantic_cluster=SemanticClustering(
        EmbeddingFactory('openai').get(), 'kmeans'
    ))
    yield compressor


def test_cluster_time_compressor_split_sentences():
    test_passage = Passage(
        content="Test is good. Test is bad. But! Test is good. Nevertheless, test is bad. But why?",
        filepath="test_filepath",
    )
    split_passages = list(ClusterTimeCompressor._split_sentences([test_passage]))
    assert len(list(split_passages)) == 5
    assert split_passages[0].content == "Test is good"
    assert split_passages[1].content == "Test is bad"
    assert split_passages[2].content == "But! Test is good"
    assert split_passages[3].content == "Nevertheless, test is bad"
    assert split_passages[4].content == "But why?"
