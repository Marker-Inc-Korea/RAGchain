import os

import pytest

from pymongo.errors import BulkWriteError
from RAGchain.DB import MongoDB
from test_base_db import TEST_PASSAGES, fetch_test_base, search_test_base, duplicate_id_test_base


@pytest.fixture(scope='module')
def mongo_db():
    assert os.getenv('MONGO_COLLECTION_NAME') == 'test'
    mongo_db = MongoDB(
        mongo_url=os.getenv('MONGO_URL'),
        db_name=os.getenv('MONGO_DB_NAME'),
        collection_name=os.getenv('MONGO_COLLECTION_NAME'))
    mongo_db.create_or_load()
    mongo_db.save(TEST_PASSAGES)
    yield mongo_db
    mongo_db.collection.drop()
    assert mongo_db.collection_name not in mongo_db.db.list_collection_names()


def test_create_or_load(mongo_db):
    assert mongo_db.collection_name in mongo_db.db.list_collection_names()


def test_fetch(mongo_db):
    fetch_test_base(mongo_db)


def test_db_type(mongo_db):
    assert mongo_db.db_type == 'mongo_db'


def test_search(mongo_db):
    search_test_base(mongo_db)


def test_duplicate_id(mongo_db):
    duplicate_id_test_base(mongo_db, BulkWriteError)
