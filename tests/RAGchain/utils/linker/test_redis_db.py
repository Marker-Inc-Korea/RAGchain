import pytest

from RAGchain.utils.linker import RedisDBSingleton
import test_linker_base


@pytest.fixture
def redis_db():
    redis_db = RedisDBSingleton()
    redis_db.put_json(test_linker_base.TEST_IDS[0], test_linker_base.TEST_DB_ORIGIN)
    yield redis_db
    redis_db.flush_db()
    assert redis_db.connection_check() is True


def test_get_json(redis_db):
    assert redis_db.get_json(test_linker_base.TEST_IDS) == [test_linker_base.TEST_DB_ORIGIN]
