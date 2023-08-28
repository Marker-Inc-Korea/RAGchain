import os

import pytest

import test_base_retrieval
from KoPrivateGPT.retrieval import BM25Retrieval


@pytest.fixture
def bm25_retrieval():
    bm25_path = os.path.join(test_base_retrieval.root_dir, "resources", "bm25", "bm25_db_retrieval.pkl")
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    yield bm25_retrieval
    os.remove(bm25_path)


def test_bm25_retrieval(bm25_retrieval):
    bm25_retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 6
    retrieved_ids = bm25_retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_ids(retrieved_ids, top_k)

    # TODO : test retrieve method after making DB linker
