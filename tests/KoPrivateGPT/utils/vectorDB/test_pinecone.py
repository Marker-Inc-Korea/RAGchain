import os

import pytest

import test_base_vectordb
from KoPrivateGPT.utils.vectorDB import Pinecone


@pytest.fixture
def pinecone():
    pinecone_namespace = "vectordb-test-namespace"
    pinecone_index_name = "vectordb-test"
    pinecone_dimension = 3
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'),
                        index_name=pinecone_index_name, namespace=pinecone_namespace, dimension=pinecone_dimension)
    yield pinecone
    pinecone.delete([vec.passage_id for vec in test_base_vectordb.TEST_VECTORS])


def test_pinecone_dby_type(pinecone):
    assert pinecone.get_db_type() == "pinecone"


def test_pinecone(pinecone):
    pinecone.add_vectors(test_base_vectordb.TEST_VECTORS)
    top_k = 2
    ids, scores = pinecone.similarity_search(query_vectors=[0.4, 0.5, 0.7], top_k=top_k)
    assert len(ids) == top_k
    assert len(scores) == top_k
    assert ids[0] in [vec.passage_id for vec in test_base_vectordb.TEST_VECTORS]
    assert ids[1] in [vec.passage_id for vec in test_base_vectordb.TEST_VECTORS]
