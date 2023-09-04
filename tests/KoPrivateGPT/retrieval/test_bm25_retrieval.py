import os

import pytest

import test_base_retrieval
from KoPrivateGPT.retrieval import BM25Retrieval


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
    # test delete
    bm25_retrieval.delete(retrieved_ids)
    retrieved_all_ids = bm25_retrieval.retrieve_id(query='What is visconde structure?',
                                                   top_k=len(test_base_retrieval.TEST_PASSAGES) - top_k)
    assert len(retrieved_all_ids) == len(test_base_retrieval.TEST_PASSAGES) - top_k
    for deleted_id in retrieved_ids:
        assert deleted_id not in retrieved_all_ids
