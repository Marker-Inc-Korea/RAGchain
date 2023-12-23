import pytest

from RAGchain.utils.linker import DynamoDBSingleton

TEST_IDS = ['test_id_1']

TEST_DB_ORIGIN = {
    'db_type': 'test_db',
    'db_path': {
        'url': 'test_host',
        'db_name': 'test_port',
        'collection_name': 'test_db_name',
    }
}


@pytest.fixture
def dynamo_db():
    dynamo_db = DynamoDBSingleton()
    dynamo_db.put_json(TEST_IDS[0], TEST_DB_ORIGIN)
    yield dynamo_db
    dynamo_db.flush_db()


def test_get_json(dynamo_db):
    assert dynamo_db.get_json(TEST_IDS) == [TEST_DB_ORIGIN]
