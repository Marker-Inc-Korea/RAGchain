import os

import pytest

import test_base_retrieval
from KoPrivateGPT.retrieval import HyDERetrieval, BM25Retrieval


@pytest.fixture
def hyde_retrieval():
    test_prompt = "Please write a scientific paper passage to answer the question"
    bm25_path = os.path.join(test_base_retrieval.root_dir, "resources", "bm25", "bm25_db_retrieval.pkl")
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    hyde_retrieval = HyDERetrieval(bm25_retrieval, system_prompt=test_prompt)
    yield hyde_retrieval
    os.remove(bm25_path)


def test_hyde_retrieval(hyde_retrieval):
    hyde_retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 4
    retrieved_ids = hyde_retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k,
                                               model_kwargs={'max_tokens': 64})
    test_base_retrieval.validate_ids(retrieved_ids, top_k)

    # TODO : test retrieve method after making DB linker
