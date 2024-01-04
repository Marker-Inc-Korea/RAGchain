import pytest
from langchain_core.runnables import RunnableLambda

from RAGchain.reranker.importance import WeightedImportanceReranker
from RAGchain.schema import Passage, RetrievalResult

TEST_PASSAGES = [
    Passage(id=str(i), content=str(i), filepath='test', importance=i) for i in range(5)
]
SCORES = [i ** 2 for i in range(5, 0, -1)]


@pytest.fixture
def weighted_importance_reranker():
    reranker = WeightedImportanceReranker(importance_weight=0.8)
    yield reranker


def test_weighted_importance_reranker(weighted_importance_reranker):
    reranked_passages = weighted_importance_reranker.rerank(TEST_PASSAGES, SCORES)
    assert isinstance(reranked_passages[0], Passage)
    solution = [4, 3, 2, 1, 0]
    for passage, idx in zip(reranked_passages, solution):
        assert passage.id == str(idx)


def test_weighted_importance_reranker_runnable(weighted_importance_reranker):
    runnable = weighted_importance_reranker | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(RetrievalResult(query="query", passages=TEST_PASSAGES, scores=SCORES))
    assert isinstance(result['passages'], list)
    assert isinstance(result['scores'], list)
    assert isinstance(result['passages'][0], Passage)
    assert isinstance(result['scores'][0], float)
    assert len(result['passages']) == len(TEST_PASSAGES)
    assert len(result['scores']) == len(result['passages'])
    solution = [4, 3, 2, 1, 0]
    for passage, idx in zip(result['passages'], solution):
        assert passage.id == str(idx)
    for i in range(1, len(result['scores'])):
        assert result['scores'][i - 1] >= result['scores'][i]
