import pytest

import test_base_reranker
from RAGchain.reranker import MonoT5Reranker

test_passages = test_base_reranker.TEST_PASSAGES[:20]
query = "What is query decomposition?"


@pytest.fixture
def mono_t5_reranker():
    reranker = MonoT5Reranker()
    yield reranker


def test_mono_t5_reranker(mono_t5_reranker):
    rerank_passages = mono_t5_reranker.rerank(query, test_passages)
    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]


def test_mono_t5_reranker_runnable(mono_t5_reranker):
    test_base_reranker.base_runnable_test(mono_t5_reranker)
