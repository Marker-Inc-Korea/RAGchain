import pytest

import test_base_linker
from RAGchain.utils.linker import RedisLinker

TEST_UUID_IDS = test_base_linker.TEST_UUID_IDS
TEST_UUID_STR_IDS = test_base_linker.TEST_UUID_STR_IDS
TEST_STR_IDS = test_base_linker.TEST_STR_IDS
TEST_DB_ORIGIN = test_base_linker.TEST_DB_ORIGIN


@pytest.fixture
def redis_db():
    redis_db = RedisLinker(allow_multiple_instances=True)
    yield redis_db
    redis_db.flush_db()
    assert redis_db.connection_check() is True


def test_get_json(redis_db):
    test_base_linker.get_json_test(redis_db, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS)


def test_no_id_warning(redis_db):
    test_base_linker.no_id_warning_test(redis_db)


def test_no_data_warning(redis_db):
    test_base_linker.no_data_warning_test(redis_db)


def test_no_data_warning2(redis_db):
    test_base_linker.no_data_warning_test2(redis_db)


def test_delete(redis_db):
    test_base_linker.delete_test(redis_db)
