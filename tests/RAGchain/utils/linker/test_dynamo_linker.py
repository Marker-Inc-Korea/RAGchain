import pytest

from RAGchain.utils.linker import DynamoDBSingleton
import test_base_linker


@pytest.fixture
def dynamo_db():
    dynamo_db = DynamoDBSingleton(allow_multiple_instances=True)
    dynamo_db.put_json(test_base_linker.TEST_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    yield dynamo_db
    dynamo_db.flush_db()


def test_get_json(dynamo_db):
    assert dynamo_db.get_json(test_base_linker.TEST_IDS) == [test_base_linker.TEST_DB_ORIGIN]
