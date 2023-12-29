import pytest

import test_base_reranker
from RAGchain.reranker.time import SimpleTimeReranker


@pytest.fixture
def simple_time_reranker():
    reranker = SimpleTimeReranker()
    yield reranker


def test_simple_time_reranker(simple_time_reranker):
    rerank_passages = simple_time_reranker.rerank(test_base_reranker.TEST_PASSAGES)
    assert len(rerank_passages) == len(test_base_reranker.TEST_PASSAGES)
    for i in range(1, len(rerank_passages)):
        assert rerank_passages[i].content_datetime <= rerank_passages[i - 1].content_datetime
