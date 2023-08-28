import os
import pathlib
import shutil

import pytest

import test_base_vectordb
from KoPrivateGPT.utils.vectorDB import Chroma


@pytest.fixture
def chroma():
    root_dir = pathlib.PurePath(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
    chroma_dir = os.path.join(root_dir, "resources", "chroma_vectordb_test")
    if not os.path.exists(chroma_dir):
        os.makedirs(chroma_dir)
    assert os.path.exists(chroma_dir)
    assert os.path.isdir(chroma_dir)
    collection_name = "test"
    chroma = Chroma(persist_dir=chroma_dir, collection_name=collection_name)
    yield chroma
    # delete all files under chroma_dir
    chroma.delete_all()
    shutil.rmtree(chroma_dir)


def test_chroma_db_type(chroma):
    assert chroma.get_db_type() == "chroma"


def test_chroma(chroma):
    chroma.add_vectors(test_base_vectordb.TEST_VECTORS)
    top_k = 2
    ids, scores = chroma.similarity_search(query_vectors=[0.4, 0.5, 0.7], top_k=top_k)
    assert len(ids) == top_k
    assert len(scores) == top_k
    assert ids[0] in [vec.passage_id for vec in test_base_vectordb.TEST_VECTORS]
    assert ids[1] in [vec.passage_id for vec in test_base_vectordb.TEST_VECTORS]
