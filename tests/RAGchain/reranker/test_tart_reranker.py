import pytest

import test_base_reranker
from RAGchain.reranker import TARTReranker


@pytest.fixture
def tart_reranker():
    reranker = TARTReranker(instruction="Find passage to answer given question")
    yield reranker


def test_tart_reranker(tart_reranker):
    test_passages = test_base_reranker.TEST_PASSAGES[:20]
    query = "What is query decomposition?"
    rerank_passages = tart_reranker.rerank(query, test_passages)
    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]
