import pytest

from RAGchain.utils.linker import RedisDBSingleton
import test_base_linker


@pytest.fixture
def redis_db():
    redis_db = RedisDBSingleton(allow_multiple_instances=True)
    redis_db.put_json(test_base_linker.TEST_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    yield redis_db
    redis_db.flush_db()
    assert redis_db.connection_check() is True


def test_get_json(redis_db):
    assert redis_db.get_json(test_base_linker.TEST_IDS) == [test_base_linker.TEST_DB_ORIGIN]
