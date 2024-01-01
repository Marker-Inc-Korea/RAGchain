from datetime import datetime, timedelta

import pytest

from RAGchain.reranker.time import WeightedTimeReranker
from RAGchain.schema import Passage

TEST_PASSAGES = [
    Passage(id=str(i), content=str(i), filepath='test', content_datetime=datetime.now() - timedelta(hours=i * 2)) for i
    in range(11)
]
SCORES = [i for i in range(11)]


@pytest.fixture
def weighted_time_reranker():
    reranker = WeightedTimeReranker(decay_rate=0.1)
    yield reranker


def test_weighted_time_reranker(weighted_time_reranker):
    reranked_passages = weighted_time_reranker.rerank(TEST_PASSAGES, SCORES)
    assert isinstance(reranked_passages[0], Passage)
    solution = [10, 9, 0, 8, 7, 1, 6, 2, 5, 3, 4]
    for passage, idx in zip(reranked_passages, solution):
        assert passage.id == str(idx)
