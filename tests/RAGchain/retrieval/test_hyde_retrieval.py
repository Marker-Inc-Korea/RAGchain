import os

import pytest

import test_base_retrieval
from RAGchain.retrieval import HyDERetrieval, BM25Retrieval


@pytest.fixture
def hyde_retrieval():
    test_prompt = "Please write a scientific paper passage to answer the question"
    bm25_path = os.path.join(test_base_retrieval.root_dir, "resources", "bm25", "test_hyde_retrieval.pkl")
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle", "test_hyde_retrieval.pkl")
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))

    test_base_retrieval.ready_pickle_db(pickle_path)
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    hyde_retrieval = HyDERetrieval(bm25_retrieval, system_prompt=test_prompt)
    yield hyde_retrieval
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    if os.path.exists(bm25_path):
        os.remove(bm25_path)


def test_hyde_retrieval(hyde_retrieval):
    hyde_retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 4
    retrieved_ids = hyde_retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k,
                                               model_kwargs={'max_tokens': 64})
    test_base_retrieval.validate_ids(retrieved_ids, top_k)
    retrieved_passages = hyde_retrieval.retrieve(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_passages(retrieved_passages, top_k)
    retrieved_ids_2, scores = hyde_retrieval.retrieve_id_with_scores(query='What is visconde structure?',
                                                                     top_k=top_k, model_kwargs={'max_tokens': 64})
    assert len(retrieved_ids_2) == len(scores)
    assert max(scores) == scores[0]
    assert min(scores) == scores[-1]


def test_hyde_retrieval_delete(hyde_retrieval):
    hyde_retrieval.ingest(test_base_retrieval.SEARCH_TEST_PASSAGES)
    hyde_retrieval.delete(['test_id_4_search', 'test_id_3_search'])
    retrieved_passages = hyde_retrieval.retrieve(query='What is visconde structure?', top_k=4)
    assert len(retrieved_passages) == 2
    assert 'test_id_1_search' in [passage.id for passage in retrieved_passages]
    assert 'test_id_2_search' in [passage.id for passage in retrieved_passages]
