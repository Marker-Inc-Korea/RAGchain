from datetime import datetime

import pytest
from langchain_core.runnables import RunnableLambda

from RAGchain.schema import Passage, RetrievalResult
from RAGchain.utils.compressor import ClusterTimeCompressor
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.semantic_clustering import SemanticClustering

TEST_PASSAGES = [
    Passage(
        id='test-1',
        content='I hear the best sports car in UK is the Corvette. Bulgogi is Korean food, and I like its taste. This framework is so cool and powerful for developing advance RAG workflow.',
        content_datetime=datetime(2021, 1, 1),
        filepath='test_filepath',
    ),
    Passage(
        id='test-2',
        content='Bulgogi is Korean food, and I think it is so delicious. I hear the best sports car in USA is the Corvette.',
        content_datetime=datetime(2021, 1, 2),
        filepath='test_filepath',
    ),
    Passage(
        id='test-3',
        content='This framework is so cool for developing RAG workflow! Corvette is the best sports car in USA and UK.',
        content_datetime=datetime(2021, 1, 3),
        filepath='test_filepath',
    )
]


@pytest.fixture
def cluster_time_compressor():
    compressor = ClusterTimeCompressor(semantic_cluster=SemanticClustering(
        EmbeddingFactory('openai').get(), 'kmeans'
    ), split_by_sentences=True)
    yield compressor


def test_cluster_time_compressor_split_sentences():
    test_passage = Passage(
        content="Test is good. Test is bad. But! Test is good. Nevertheless, test is bad. But why? But, I don\'t know my feelings now.",
        filepath="test_filepath",
    )
    split_passages = list(ClusterTimeCompressor._split_sentences([test_passage]))
    assert len(list(split_passages)) == 7
    assert split_passages[0].content == "Test is good."
    assert split_passages[1].content == "Test is bad."
    assert split_passages[2].content == "But!"
    assert split_passages[3].content == "Test is good."
    assert split_passages[4].content == "Nevertheless, test is bad."
    assert split_passages[5].content == "But why?"
    assert split_passages[6].content == "But, I don\'t know my feelings now."


def test_cluster_time_compressor(cluster_time_compressor):
    compressed_passages = cluster_time_compressor.compress(TEST_PASSAGES, n_clusters=3)
    assert len(compressed_passages) == 3
    ids = [passage.id for passage in compressed_passages]
    assert 'test-1' not in ids


def test_cluster_time_compressor_runnable(cluster_time_compressor):
    runnable = cluster_time_compressor | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(RetrievalResult(
        query='test',
        passages=TEST_PASSAGES,
        scores=[1.0, 0.9, 0.8],
    ), config={"configurable": {"compressor_options": {"n_clusters": 3}}})
    assert len(result['passages']) == 3
    assert 'test-1' not in [passage.id for passage in result['passages']]
    assert len(result['scores']) == 0
