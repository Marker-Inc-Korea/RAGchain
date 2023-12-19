import logging
import os
import shutil

import chromadb
import pytest
from langchain.vectorstores import Chroma

import test_base_retrieval
from RAGchain.DB import PickleDB
from RAGchain.retrieval import BM25Retrieval, VectorDBRetrieval, HybridRetrieval
from RAGchain.utils.embed import EmbeddingFactory

logger = logging.getLogger(__file__)


@pytest.fixture(scope='module')
def hybrid_retrieval():
    bm25_path = os.path.join(test_base_retrieval.root_dir, "resources", "bm25", "test_hybrid_retrieval.pkl")
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle", "test_hybrid_retrieval.pkl")
    chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "test_hybrid_retrieval_chroma")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))

    test_base_retrieval.ready_pickle_db(pickle_path)
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    vectordb_retrieval = VectorDBRetrieval(vectordb=Chroma(
        client=chromadb.PersistentClient(path=chroma_path),
        collection_name='test_hybrid_retrieval',
        embedding_function=EmbeddingFactory('openai').get()
    ))
    hybrid_retrieval = HybridRetrieval(retrievals=[bm25_retrieval, vectordb_retrieval], weights=[0.3, 0.7], p=50)
    yield hybrid_retrieval
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


@pytest.fixture
def hybrid_retrieval_for_delete():
    bm25_path = os.path.join(test_base_retrieval.root_dir, "resources", "bm25", "test_hybrid_retrieval_for_delete.pkl")
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle",
                               "test_hybrid_retrieval_for_delete.pkl")
    chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "test_hybrid_retrieval_for_delete_chroma")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))

    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(test_base_retrieval.SEARCH_TEST_PASSAGES)

    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    vectordb_retrieval = VectorDBRetrieval(vectordb=Chroma(
        client=chromadb.PersistentClient(path=chroma_path),
        collection_name='test_hybrid_retrieval',
        embedding_function=EmbeddingFactory('openai').get()
    ))
    hybrid_retrieval = HybridRetrieval(retrievals=[bm25_retrieval, vectordb_retrieval], weights=[0.3, 0.7], p=50)
    hybrid_retrieval.ingest(test_base_retrieval.SEARCH_TEST_PASSAGES)
    yield hybrid_retrieval
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


def test_hybrid_retrieval(hybrid_retrieval):
    hybrid_retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 6
    retrieved_ids = hybrid_retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_ids(retrieved_ids, top_k)
    retrieved_passages = hybrid_retrieval.retrieve(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_passages(retrieved_passages, top_k)
    retrieved_ids_2, scores = hybrid_retrieval.retrieve_id_with_scores(query='What is visconde structure?',
                                                                       top_k=top_k)
    logger.info(f'scores: {scores}')
    assert retrieved_ids == retrieved_ids_2
    assert len(retrieved_ids_2) == len(scores)
    assert max(scores) == scores[0]
    assert min(scores) == scores[-1]


def test_hybrid_retrieval_rrf(hybrid_retrieval):
    hybrid_retrieval.method = 'rrf'
    test_hybrid_retrieval(hybrid_retrieval)


def test_hybrid_retrieval_delete(hybrid_retrieval_for_delete):
    hybrid_retrieval_for_delete.delete(['test_id_4_search', 'test_id_3_search'])
    retrieved_passages = hybrid_retrieval_for_delete.retrieve(query='What is visconde structure?', top_k=4)
    assert len(retrieved_passages) == 2
    assert 'test_id_1_search' in [passage.id for passage in retrieved_passages]
    assert 'test_id_2_search' in [passage.id for passage in retrieved_passages]
