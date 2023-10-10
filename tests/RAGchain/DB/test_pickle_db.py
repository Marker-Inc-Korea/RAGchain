import os
import pathlib

import pytest

from RAGchain.DB import PickleDB
from test_base_db import fetch_test_base, TEST_PASSAGES, search_test_base


@pytest.fixture(scope='module')
def pickle_db():
    root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
    resource_dir = os.path.join(root_dir, "resources")
    pickle_db_path = os.path.join(resource_dir, "pickle", "pickle_db.pkl")
    pickle_db = PickleDB(
        save_path=pickle_db_path
    )
    pickle_db.create_or_load()
    pickle_db.save(TEST_PASSAGES)
    yield pickle_db
    os.remove(pickle_db_path)


def test_create_or_load(pickle_db):
    assert os.path.exists(os.path.dirname(pickle_db.save_path))


def test_fetch(pickle_db):
    fetch_test_base(pickle_db)


def test_db_type(pickle_db):
    assert pickle_db.db_type == 'pickle_db'


def test_search(pickle_db):
    search_test_base(pickle_db)
