import os

import pytest

import test_base_db
from RAGchain.DB import MongoDB


@pytest.fixture(scope='module')
def mongo_db():
    assert os.getenv('MONGO_COLLECTION_NAME') == 'test'
    mongo_db = MongoDB(
        mongo_url=os.getenv('MONGO_URL'),
        db_name=os.getenv('MONGO_DB_NAME'),
        collection_name=os.getenv('MONGO_COLLECTION_NAME'))
    mongo_db.create_or_load()
    # drop first, as any leftover garbage data in mongodb may cause the test to fail.
    mongo_db.collection.drop()
    mongo_db.save(test_base_db.TEST_PASSAGES)
    yield mongo_db
    mongo_db.collection.drop()
    assert mongo_db.collection_name not in mongo_db.db.list_collection_names()


def test_create_or_load(mongo_db):
    assert mongo_db.collection_name in mongo_db.db.list_collection_names()


def test_fetch(mongo_db):
    test_base_db.fetch_test_base(mongo_db)


def test_db_type(mongo_db):
    assert mongo_db.db_type == 'mongo_db'


def test_search(mongo_db):
    test_base_db.search_test_base(mongo_db)
