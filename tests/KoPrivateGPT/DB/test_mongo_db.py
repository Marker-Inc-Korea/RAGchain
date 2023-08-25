import os

import pytest

import test_base
from KoPrivateGPT.DB import MongoDB


@pytest.fixture(scope='module')
def mongo_db():
    assert os.getenv('MONGO_COLLECTION_NAME') == 'test'
    mongo_db = MongoDB(
        mongo_url=os.getenv('MONGO_URL'),
        db_name=os.getenv('MONGO_DB_NAME'),
        collection_name=os.getenv('MONGO_COLLECTION_NAME'))
    mongo_db.create_or_load()
    yield mongo_db
    mongo_db.collection.drop()
    assert mongo_db.collection_name not in mongo_db.db.list_collection_names()


def test_create_or_load(mongo_db):
    assert mongo_db.collection_name in mongo_db.db.list_collection_names()


def test_fetch(mongo_db):
    test_base.fetch_test_base(mongo_db)


def test_db_type(mongo_db):
    assert mongo_db.db_type == 'mongo_db'


def test_search(mongo_db):
    test_result_1 = mongo_db.search({'filepath': './test/second_file.txt'})
    assert len(test_result_1) == 2
    assert 'test_id_2' in [passage.id for passage in test_result_1]
    assert 'test_id_3' in [passage.id for passage in test_result_1]

    test_result_2 = mongo_db.search({'metadata_etc.test': 'test1'})
    assert len(test_result_2) == 1
    assert 'test_id_1' == test_result_2[0].id
