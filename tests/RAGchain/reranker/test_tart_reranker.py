import pytest
from langchain_core.runnables import RunnableLambda

import test_base_reranker
from RAGchain.reranker import TARTReranker
from RAGchain.schema import RetrievalResult

test_passages = test_base_reranker.TEST_PASSAGES[:20]
query = "What is query decomposition?"


@pytest.fixture
def tart_reranker():
    reranker = TARTReranker(instruction="Find passage to answer given question")
    yield reranker


def test_tart_reranker(tart_reranker):
    rerank_passages = tart_reranker.rerank(query, test_passages)
    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]


def test_tart_reranker_runnable(tart_reranker):
    runnable = tart_reranker | RunnableLambda(lambda x: x.to_dict())
    rerank_passages = runnable.invoke(RetrievalResult(query=query, passages=test_passages, scores=[]))
    assert len(rerank_passages['passages']) == len(test_passages)
    assert rerank_passages['passages'][0] != test_passages[0] or rerank_passages['passages'][-1] != test_passages[-1]
    assert rerank_passages['query'] == query
    for i in range(1, len(rerank_passages['scores'])):
        assert rerank_passages['scores'][i - 1] >= rerank_passages['scores'][i]
