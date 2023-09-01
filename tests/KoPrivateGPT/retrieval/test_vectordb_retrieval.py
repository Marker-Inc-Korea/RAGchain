import os
import shutil

import pytest

import test_base_retrieval
from KoPrivateGPT.retrieval import VectorDBRetrieval
from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.vectorDB import Chroma


@pytest.fixture
def vectordb_retrieval():
    pickle_path = os.path.join(test_base_retrieval.root_dir, "resources", "pickle", "test_vectordb_retrieval.pkl")
    if not os.path.exists(os.path.dirname(pickle_path)):
        os.makedirs(os.path.dirname(pickle_path))
    test_base_retrieval.ready_pickle_db(pickle_path)
    chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "test_vectordb_retrieval_chroma")
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
    chroma = Chroma(persist_dir=chroma_path, collection_name='test_vectordb_retrieval')
    retrieval = VectorDBRetrieval(vectordb=chroma, embedding=EmbeddingFactory('openai').get())
    yield retrieval
    # teardown
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    chroma.delete_all()
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


def test_vectordb_retrieval(vectordb_retrieval):
    vectordb_retrieval.ingest(test_base_retrieval.TEST_PASSAGES)
    top_k = 6
    retrieved_ids = vectordb_retrieval.retrieve_id(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_ids(retrieved_ids, top_k)
    retrieved_passages = vectordb_retrieval.retrieve(query='What is visconde structure?', top_k=top_k)
    test_base_retrieval.validate_passages(retrieved_passages, top_k)
