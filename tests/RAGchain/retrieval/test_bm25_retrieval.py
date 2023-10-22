import os

import pytest

import test_base_retrieval
from RAGchain.retrieval import BM25Retrieval


@pytest.fixture
def bm25_retrieval():
    bm25_path = os.path.join(test_base_retrieval.root_dir, "resources", "bm25", "test_bm25_retrieval.pkl")
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle", "test_bm25_retrieval.pkl")
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    test_base_retrieval.ready_pickle_db(pickle_path)
    yield bm25_retrieval
    # teardown
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_bm25_retrieval(bm25_retrieval):
    bm25_retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 6
    retrieved_ids = bm25_retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_ids(retrieved_ids, top_k)
    retrieved_passages = bm25_retrieval.retrieve(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_passages(retrieved_passages, top_k)
    retrieved_ids_2, scores = bm25_retrieval.retrieve_id_with_scores(query='What is visconde structure?',
                                                                     top_k=top_k)
    assert retrieved_ids == retrieved_ids_2
    assert len(retrieved_ids_2) == len(scores)
    assert max(scores) == scores[0]
    assert min(scores) == scores[-1]

    bm25_retrieval.ingest(test_base_retrieval.SEARCH_TEST_PASSAGES)
    retrieved_passages = bm25_retrieval.retrieve_with_filter(
        query='What is visconde structure?',
        top_k=top_k,
        content=['This is test number 1', 'This is test number 3']
    )
    assert len(retrieved_passages) == 3
    assert 'test_id_1_search' in [passage.id for passage in retrieved_passages]


def test_bm25_retrieval_delete(bm25_retrieval):
    bm25_retrieval.ingest(test_base_retrieval.SEARCH_TEST_PASSAGES)
    bm25_retrieval.delete(['test_id_4_search', 'test_id_3_search'])
    retrieved_passages = bm25_retrieval.retrieve(query='What is visconde structure?', top_k=4)
    assert len(retrieved_passages) == 2
    assert 'test_id_1_search' in [passage.id for passage in retrieved_passages]
    assert 'test_id_2_search' in [passage.id for passage in retrieved_passages]
