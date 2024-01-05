import pytest
from langchain_core.runnables import RunnableLambda

from RAGchain.reranker.importance import SimpleImportanceReranker
from RAGchain.schema import Passage, RetrievalResult

TEST_PASSAGES = [
    Passage(
        id='test-1',
        content="This is a test passage.",
        filepath="test.txt",
        importance=-1,
    ),
    Passage(
        id='test-2',
        content="This is a test passage.",
        filepath="test.txt",
    ),
    Passage(
        id='test-3',
        content="This is a test passage.",
        filepath="test.txt",
        importance=2,
    ),
    Passage(
        id='test-4',
        content="This is a test passage.",
        filepath="test.txt",
        importance=1,
    ),
]


@pytest.fixture
def simple_importance_reranker():
    reranker = SimpleImportanceReranker()
    yield reranker


def test_simple_importance_reranker(simple_importance_reranker):
    rerank_passages = simple_importance_reranker.rerank(TEST_PASSAGES)
    assert len(rerank_passages) == len(TEST_PASSAGES)
    assert rerank_passages[0].importance == 2
    assert rerank_passages[1].importance == 1
    assert rerank_passages[2].importance == 0
    assert rerank_passages[3].importance == -1

    assert rerank_passages[0].id == 'test-3'
    assert rerank_passages[1].id == 'test-4'
    assert rerank_passages[2].id == 'test-2'
    assert rerank_passages[3].id == 'test-1'


def test_simple_importance_reranker_runnable(simple_importance_reranker):
    runnable = simple_importance_reranker | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(RetrievalResult(query='query', passages=TEST_PASSAGES,
                                             scores=[i for i in range(len(TEST_PASSAGES))]))
    assert isinstance(result['passages'], list)
    assert isinstance(result['passages'][0], Passage)
    assert isinstance(result['scores'], list)
    assert isinstance(result['scores'][0], float)
    assert result['passages'][0].id == 'test-3'
    assert result['passages'][1].id == 'test-4'
    assert result['passages'][2].id == 'test-2'
    assert result['passages'][3].id == 'test-1'

    assert result['passages'][0].importance == 2
    assert result['passages'][1].importance == 1
    assert result['passages'][2].importance == 0
    assert result['passages'][3].importance == -1

    assert result['scores'][0] == 2
    assert result['scores'][1] == 3
    assert result['scores'][2] == 1
    assert result['scores'][3] == 0
