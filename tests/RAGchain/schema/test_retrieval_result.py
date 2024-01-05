from RAGchain.schema import RetrievalResult, Passage

TEST_PASSAGES = [Passage(
    id='test1',
    content='test1',
    filepath='test1',
), Passage(
    id='test2',
    content='test2',
    filepath='test2',
), Passage(
    id='test3',
    content='test3',
    filepath='test3',
)]

retrieval_result1 = RetrievalResult(query="test1", passages=[TEST_PASSAGES[0]], scores=[1.0])
retrieval_result2 = RetrievalResult(query="test1", passages=[TEST_PASSAGES[1]], scores=[0.5])
retrieval_result3 = RetrievalResult(query="test3", passages=[TEST_PASSAGES[2]], scores=[0.3])


def test_retrieval_result_add():
    add_result = retrieval_result1 + retrieval_result2
    assert isinstance(add_result, RetrievalResult)
    assert add_result.query == "test1"
    assert add_result.passages == [TEST_PASSAGES[0], TEST_PASSAGES[1]]
    assert add_result.scores == [1.0, 0.5]

    sum_result = sum([retrieval_result1, retrieval_result2, retrieval_result3])
    assert isinstance(sum_result, RetrievalResult)
    assert sum_result.query == "test1\ntest3"
    assert sum_result.passages == [TEST_PASSAGES[0], TEST_PASSAGES[1], TEST_PASSAGES[2]]
    assert sum_result.scores == [1.0, 0.5, 0.3]


def test_retrieval_to_dict():
    assert retrieval_result1.to_dict() == {
        "query": "test1",
        "passages": [TEST_PASSAGES[0]],
        "scores": [1.0],
        "metadata": {}
    }
