import pytest

import test_base_reranker
from RAGchain.reranker import BM25Reranker  # 빨간색을 없애라!


@pytest.fixture
def bm25_reranker():
    reranker = BM25Reranker(save_path="tests/RAGchain/reranker/test_bm25_reranker.pkl")
    yield reranker


def test_bm25_reranker(bm25_reranker):
    test_passages = test_base_reranker.TEST_PASSAGES[:20]
    query = "What is query decomposition?"
    rerank_passages = bm25_reranker.rerank(query, test_passages)
    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]

# assert rerank_passages[0] == test_passages[0] or rerank_passages[-1] == test_passages[-1] # 이렇게 하면 테스트가 통과되지 않는다.
# assert rerank_passages[0] == test_passages[0] and rerank_passages[-1] == test_passages[-1] # 이렇게 하면 테스트가 통과된다.
