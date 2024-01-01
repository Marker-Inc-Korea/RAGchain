import pytest

from RAGchain.reranker.importance import WeightedImportanceReranker
from RAGchain.schema import Passage

TEST_PASSAGES = [
    Passage(id=str(i), content=str(i), filepath='test', importance=i) for i in range(5)
]
SCORES = [i ** 2 for i in range(5, 0, -1)]


@pytest.fixture
def weighted_importance_reranker():
    reranker = WeightedImportanceReranker(importance_weight=0.8)
    yield reranker


def test_weighted_time_reranker(weighted_importance_reranker):
    reranked_passages = weighted_importance_reranker.rerank(TEST_PASSAGES, SCORES)
    assert isinstance(reranked_passages[0], Passage)
    solution = [4, 3, 2, 1, 0]
    for passage, idx in zip(reranked_passages, solution):
        assert passage.id == str(idx)
