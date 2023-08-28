import os
from uuid import UUID

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
    assert len(retrieved_ids) == top_k
    for _id in retrieved_ids:
        assert isinstance(_id, str) or isinstance(_id, UUID)
        fetch_result = list(filter(lambda x: getattr(x, 'id') == _id, test_base_retrieval.TEST_PASSAGES))
        assert len(fetch_result) == 1
        assert fetch_result[0].id == _id

    # TODO : test retrieve method after making DB linker
