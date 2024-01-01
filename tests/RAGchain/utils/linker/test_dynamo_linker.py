import pytest

from RAGchain.utils.linker import DynamoLinker, NoIdWarning, NoDataWarning

import test_base_linker

TEST_UUID_IDS = test_base_linker.TEST_UUID_IDS
TEST_UUID_STR_IDS = test_base_linker.TEST_UUID_STR_IDS
TEST_STR_IDS = test_base_linker.TEST_STR_IDS


@pytest.fixture
def dynamo_db():
    dynamo_db = DynamoLinker(allow_multiple_instances=True)
    yield dynamo_db
    dynamo_db.flush_db()


def test_get_json(dynamo_db):
    test_base_linker.get_json_test(dynamo_db, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS)


def test_no_id_warning(dynamo_db):
    test_base_linker.no_id_warning_test(dynamo_db)
