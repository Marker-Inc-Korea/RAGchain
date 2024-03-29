import pytest
from langchain_core.runnables import RunnableLambda

import test_base_reranker
from RAGchain.reranker.time import SimpleTimeReranker
from RAGchain.schema import RetrievalResult


@pytest.fixture
def simple_time_reranker():
    reranker = SimpleTimeReranker()
    yield reranker


def test_simple_time_reranker(simple_time_reranker):
    rerank_passages = simple_time_reranker.rerank(test_base_reranker.TEST_PASSAGES)
    assert len(rerank_passages) == len(test_base_reranker.TEST_PASSAGES)
    for i in range(1, len(rerank_passages)):
        assert rerank_passages[i].content_datetime <= rerank_passages[i - 1].content_datetime


def test_simple_time_reranker_runnable(simple_time_reranker):
    runnable = simple_time_reranker | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(RetrievalResult(
        query="What is reranker role?",
        passages=test_base_reranker.TEST_PASSAGES,
        scores=[i for i in range(len(test_base_reranker.TEST_PASSAGES))]
    ))
    rerank_passages = result['passages']
    assert len(rerank_passages) == len(test_base_reranker.TEST_PASSAGES)
    for i in range(1, len(rerank_passages)):
        assert rerank_passages[i].content_datetime <= rerank_passages[i - 1].content_datetime
    assert result['scores'] == [i for i in range(len(test_base_reranker.TEST_PASSAGES) - 1, -1, -1)]
