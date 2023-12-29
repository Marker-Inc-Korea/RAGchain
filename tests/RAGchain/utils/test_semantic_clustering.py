import itertools
import logging
from datetime import datetime

from RAGchain.schema import Passage
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.semantic_clustering import SemanticClustering

logger = logging.getLogger(__name__)

TEST_PASSAGES = [
    Passage(
        id='test-0',
        content='I hear the best sports car in USA is the Corvette.',
        content_datetime=datetime(2021, 1, 1),
        filepath='test_semantic_clustering'
    ),
    Passage(
        id='test-1',
        content='Bulgogi is Korean food, and I like its taste.',
        content_datetime=datetime(2021, 1, 2),
        filepath='test_semantic_clustering'
    ),
    Passage(
        id='test-2',
        content='I hear the best sports car in UK is the Corvette.',
        content_datetime=datetime(2021, 1, 3),
        filepath='test_semantic_clustering'
    ),
    Passage(
        id='test-3',
        content='Corvette is the best sports car in USA and UK.',
        content_datetime=datetime(2021, 1, 4),
        filepath='test_semantic_clustering'
    ),
    Passage(
        id='test-4',
        content='Bulgogi is Korean food, and I think it is so delicious.',
        content_datetime=datetime(2021, 1, 5),
        filepath='test_semantic_clustering'
    ),
    Passage(
        id='test-5',
        content='Wow! This framework is so cool and powerful for developing advance RAG workflow!',
        content_datetime=datetime(2021, 1, 6),
        filepath='test_semantic_clustering'
    )
]


def test_semantic_clustering():
    openai_embedding = EmbeddingFactory('openai').get()
    # works great
    for clustering_algorithm in ['kmeans', 'mean_shift', 'spectral_clustering', 'affinity_propagation']:
        semantic_clustering = SemanticClustering(openai_embedding, clustering_algorithm)
        if clustering_algorithm is 'kmeans' or clustering_algorithm is 'spectral_clustering':
            clusters = semantic_clustering.cluster(TEST_PASSAGES, n_clusters=3)
        else:
            clusters = semantic_clustering.cluster(TEST_PASSAGES)
        assert len(clusters) == 3

    # works bad
    for clustering_algorithm in ['birch', 'dbscan', 'optics', 'agglomerative_clustering']:
        semantic_clustering = SemanticClustering(openai_embedding, clustering_algorithm)
        clusters = semantic_clustering.cluster(TEST_PASSAGES)
        assert len(list(itertools.chain.from_iterable(clusters))) == 6
