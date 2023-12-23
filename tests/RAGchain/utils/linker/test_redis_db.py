import pytest

from RAGchain.utils.linker import RedisDBSingleton

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
def redis_db():
    redis_db = RedisDBSingleton()
    redis_db.put_json(TEST_IDS[0], TEST_DB_ORIGIN)
    yield redis_db
    redis_db.flush_db()
    assert redis_db.connection_check() is True


def test_get_json(redis_db):
    assert redis_db.get_json(TEST_IDS) == [TEST_DB_ORIGIN]
