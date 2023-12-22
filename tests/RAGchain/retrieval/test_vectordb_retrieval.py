import os
import shutil
from datetime import datetime

import chromadb
import pytest
from langchain.vectorstores import Chroma

import test_base_retrieval
from RAGchain.DB import PickleDB
from RAGchain.retrieval import VectorDBRetrieval
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim


@pytest.fixture(scope='module')
def vectordb_retrieval():
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle", "test_vectordb_retrieval.pkl")
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))
    test_base_retrieval.ready_pickle_db(pickle_path)
    chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "test_vectordb_retrieval_chroma")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    chroma = Chroma(client=chromadb.PersistentClient(path=chroma_path),
                    collection_name='test_vectordb_retrieval',
                    embedding_function=EmbeddingFactory('openai').get())
    retrieval = VectorDBRetrieval(vectordb=chroma)
    yield retrieval
    # teardown
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


@pytest.fixture(scope='module')
def vectordb_retrieval_for_delete():
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle",
                               "test_vectordb_retrieval_for_delete.pkl")
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))
    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(test_base_retrieval.SEARCH_TEST_PASSAGES)
    chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "test_vectordb_retrieval_for_delete_chroma")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    chroma = ChromaSlim(
        client=chromadb.PersistentClient(path=chroma_path),
        collection_name='test_vectordb_retrieval_for_delete',
        embedding_function=EmbeddingFactory('openai').get()
    )
    retrieval = VectorDBRetrieval(vectordb=chroma)
    retrieval.ingest(test_base_retrieval.SEARCH_TEST_PASSAGES)
    yield retrieval
    # teardown
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


@pytest.fixture(scope='module')
def slim_vectordb_retrieval():
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle", "test_slim_vectordb_retrieval.pkl")
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))
    test_base_retrieval.ready_pickle_db(pickle_path)
    chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "test_slim_vectordb_retrieval_chroma")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    chroma = ChromaSlim(
        client=chromadb.PersistentClient(path=chroma_path),
        collection_name='test_slim_vectordb_retrieval',
        embedding_function=EmbeddingFactory('openai').get()
    )
    retrieval = VectorDBRetrieval(vectordb=chroma)
    yield retrieval
    # teardown
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


def test_vectordb_retrieval(vectordb_retrieval):
    vectordb_retrieval_test(vectordb_retrieval)


def test_vectordb_retrieval_slim(slim_vectordb_retrieval):
    vectordb_retrieval_test(slim_vectordb_retrieval)


def vectordb_retrieval_test(retrieval: VectorDBRetrieval):
    retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 6
    retrieved_ids = retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_ids(retrieved_ids, top_k)
    retrieved_passages = retrieval.retrieve(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_passages(retrieved_passages, top_k)
    retrieved_ids_2, scores = retrieval.retrieve_id_with_scores(query='What is visconde structure?',
                                                                top_k=top_k)
    assert len(retrieved_ids_2) == len(scores)
    assert max(scores) == scores[0]
    assert min(scores) == scores[-1]

    retrieval.ingest(test_base_retrieval.SEARCH_TEST_PASSAGES)
    retrieved_passages = retrieval.retrieve_with_filter(
        query='What is visconde structure?',
        top_k=top_k,
        content=['This is test number 1', 'This is test number 3'],
        content_datetime_range=[(datetime(2020, 12, 1), datetime(2021, 1, 31))]
    )
    assert len(retrieved_passages) == 2
    assert 'test_id_1_search' in [passage.id for passage in retrieved_passages]
    assert 'test_id_3_search' in [passage.id for passage in retrieved_passages]


def test_vectordb_retrieval_delete(vectordb_retrieval_for_delete):
    vectordb_retrieval_for_delete.delete(['test_id_4_search', 'test_id_3_search'])
    retrieved_passages = vectordb_retrieval_for_delete.retrieve(query='What is visconde structure?', top_k=4)
    assert len(retrieved_passages) == 2
    assert 'test_id_1_search' in [passage.id for passage in retrieved_passages]
    assert 'test_id_2_search' in [passage.id for passage in retrieved_passages]
