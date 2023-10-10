import os
import pathlib
from typing import List

import pytest
from langchain.schema import Document

from RAGchain.DB import PickleDB
from RAGchain.schema import Passage
from RAGchain.utils.file_cache import FileCache

test_passages: List[Passage] = [
    Passage(content="test1", filepath="test1"),
    Passage(content="test2", filepath="test2"),
    Passage(content="test3", filepath="test2")
]

test_documents: List[Document] = [
    Document(page_content="ttt1211", metadata={"source": "test1"}),
    Document(page_content="asdf", metadata={"source": "test2"}),
    Document(page_content="hgh", metadata={"source": "test3"}),
    Document(page_content="egrgfg", metadata={"source": "test4"}),
    Document(page_content="hhhh", metadata={"source": "test4"}),
]


@pytest.fixture
def file_cache():
    root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
    pickle_path = os.path.join(root_dir, "resources", "pickle", "test_file_cache.pkl")
    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(test_passages)
    file_cache = FileCache(db)
    yield file_cache
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_file_cache(file_cache):
    result_documents = file_cache.delete_duplicate(test_documents)
    assert len(result_documents) == 3
    for doc in result_documents:
        assert doc.metadata['source'] != 'test1' and doc.metadata['source'] != 'test2'
