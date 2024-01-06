import pytest

import test_base_reranker
from RAGchain.reranker import UPRReranker

test_passages = test_base_reranker.TEST_PASSAGES[:20]
query = "What is query decomposition?"


@pytest.fixture
def upr_reranker():
    reranker = UPRReranker()
    yield reranker


def test_upr_reranker(upr_reranker):
    rerank_passages = upr_reranker.rerank(query, test_passages)
    assert len(rerank_passages) == len(test_passages)
    assert rerank_passages[0] != test_passages[0] or rerank_passages[-1] != test_passages[-1]


def test_calculate_likelihood(upr_reranker):
    question = "Who is the most popular girl group in South Korea?"
    contexts = ["The ironman in the Marvel movie once fought with Captain America.",
                "New Jeans is the most popular girl group in South Korea.",
                "Pizza is Italian food. It is made of flour, tomato sauce, and cheese."]
    indexes, scores = upr_reranker.calculate_likelihood(question, contexts)
    assert indexes[0] == 1
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_upr_reranker_runnable(upr_reranker):
    test_base_reranker.base_runnable_test(upr_reranker)
