import pytest

import test_base_reranker
from RAGchain.reranker import BM25Reranker
from RAGchain.schema import Passage


@pytest.fixture
def bm25_reranker():
    reranker = BM25Reranker()
    yield reranker


def test_bm25_reranker(bm25_reranker):
    test_passages = test_base_reranker.TEST_PASSAGES[:20]
    query = "What is query decomposition?"
    rerank_passages = bm25_reranker.rerank(query, test_passages)

    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]
    assert isinstance(rerank_passages[0], Passage)
