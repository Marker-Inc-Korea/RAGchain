from datetime import datetime, timedelta

import pytest
from langchain_core.runnables import RunnableLambda

from RAGchain.reranker.time import WeightedTimeReranker
from RAGchain.schema import Passage, RetrievalResult

TEST_PASSAGES = [
    Passage(id=str(i), content=str(i), filepath='test', content_datetime=datetime.now() - timedelta(hours=i * 2)) for i
    in range(11)
]
SCORES = [i for i in range(11)]


@pytest.fixture
def weighted_time_reranker():
    reranker = WeightedTimeReranker(decay_rate=0.1)
    yield reranker


def test_weighted_time_reranker(weighted_time_reranker):
    reranked_passages = weighted_time_reranker.rerank(TEST_PASSAGES, SCORES)
    assert isinstance(reranked_passages[0], Passage)
    solution = [10, 9, 0, 8, 7, 1, 6, 2, 5, 3, 4]
    for passage, idx in zip(reranked_passages, solution):
        assert passage.id == str(idx)


def test_weighted_time_reranker_runnable(weighted_time_reranker):
    runnable = weighted_time_reranker | {
        "passages": RunnableLambda(lambda x: x.passages),
        "scores": RunnableLambda(lambda x: x.scores)
    }

    result = runnable.invoke(RetrievalResult(query="query", passages=TEST_PASSAGES, scores=SCORES))
    assert isinstance(result['passages'], list)
    assert isinstance(result['scores'], list)
    assert isinstance(result['passages'][0], Passage)
    assert isinstance(result['scores'][0], float)
    solution = [10, 9, 0, 8, 7, 1, 6, 2, 5, 3, 4]
    for passage, idx in zip(result['passages'], solution):
        assert passage.id == str(idx)
    for i in range(1, len(result['scores'])):
        assert result['scores'][i - 1] >= result['scores'][i]
