import time

import pytest

import test_base_linker
from RAGchain.utils.linker import DynamoLinker

TEST_UUID_IDS = test_base_linker.TEST_UUID_IDS
TEST_UUID_STR_IDS = test_base_linker.TEST_UUID_STR_IDS
TEST_STR_IDS = test_base_linker.TEST_STR_IDS


@pytest.fixture
def dynamo_db():
    dynamo_db = DynamoLinker(allow_multiple_instances=True)
    yield dynamo_db
    dynamo_db.flush_db()
    time.sleep(5)


def test_get_json(dynamo_db):
    test_base_linker.get_json_test(dynamo_db, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS)


def test_no_id_warning(dynamo_db):
    test_base_linker.no_id_warning_test(dynamo_db)


def test_no_data_warning(dynamo_db):
    test_base_linker.no_data_warning_test(dynamo_db)


def test_no_data_warning2(dynamo_db):
    test_base_linker.no_data_warning_test2(dynamo_db)


def test_delete(dynamo_db):
    test_base_linker.delete_test(dynamo_db)


def test_long(dynamo_db):
    test_base_linker.long_test(dynamo_db)


def test_long_26(dynamo_db):
    test_base_linker.long_26_test(dynamo_db)
