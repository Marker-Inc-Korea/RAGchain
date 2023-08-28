import os
import pathlib

import pytest

from KoPrivateGPT.DB import PickleDB
from test_base_db import fetch_test_base, TEST_PASSAGES


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
    test_result_1 = pickle_db.search({'filepath': './test/second_file.txt'})
    assert len(test_result_1) == 2
    assert 'test_id_2' in [passage.id for passage in test_result_1]
    assert 'test_id_3' in [passage.id for passage in test_result_1]

    test_result_2 = pickle_db.search({'test': 'test1'})
    assert len(test_result_2) == 1
    assert 'test_id_1' == test_result_2[0].id
