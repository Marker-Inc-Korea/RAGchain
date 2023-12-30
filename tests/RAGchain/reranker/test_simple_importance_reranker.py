import pytest

from RAGchain.reranker.importance import SimpleImportanceReranker
from RAGchain.schema import Passage

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
