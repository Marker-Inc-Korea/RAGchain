import pytest
from langchain_core.runnables import RunnableLambda

import test_base_reranker
from RAGchain.reranker import MonoT5Reranker
from RAGchain.schema import RetrievalResult

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
    runnable = mono_t5_reranker | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(RetrievalResult(
        query=query,
        passages=test_passages,
        scores=[],
    ))
    assert len(result['passages']) == len(test_passages)
    assert result['passages'][0] != test_passages[0] or result['passages'][-1] != test_passages[-1]
    assert len(result['passages']) == len(result['scores'])
    for i in range(1, len(result['scores'])):
        assert result['scores'][i - 1] >= result['scores'][i]
