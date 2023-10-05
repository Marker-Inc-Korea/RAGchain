import pytest

import test_base_reranker
from KoPrivateGPT.reranker import LLMReranker


@pytest.fixture
def llm_reranker():
    reranker = LLMReranker()
    yield reranker


def test_llm_reranker(llm_reranker):
    test_passages = test_base_reranker.TEST_PASSAGES[:20]
    query = "What is query decomposition?"
    rerank_passages = llm_reranker.rerank(query, test_passages)
    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]

    window_rerank_passages = llm_reranker.rerank_sliding_window(query, test_passages, 5)
    assert len(window_rerank_passages) == len(test_passages)
    assert window_rerank_passages[0] != test_passages[0] or window_rerank_passages[-1] != test_passages[-1]
