import pytest

from RAGchain.utils.linker import RedisLinker, NoIdWarning, NoDataWarning

import test_base_linker

TEST_UUID_IDS = test_base_linker.TEST_UUID_IDS
TEST_UUID_STR_IDS = test_base_linker.TEST_UUID_STR_IDS
TEST_STR_IDS = test_base_linker.TEST_STR_IDS


@pytest.fixture
def redis_db():
    redis_db = RedisLinker(allow_multiple_instances=True)
    yield redis_db
    redis_db.flush_db()
    assert redis_db.connection_check() is True


def test_get_json(redis_db):
    test_base_linker.get_json_test(redis_db, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS)


def test_no_id_warning(redis_db):
    with pytest.warns(NoIdWarning) as record:
        redis_db.get_json(['fake_id'])
    assert "ID fake_id not found in RedisLinker" in str(record[0].message)


def test_no_data_warning(redis_db):
    redis_db.put_json(test_base_linker.TEST_STR_IDS[0], None)
    with pytest.warns(NoDataWarning) as record:
        redis_db.get_json(test_base_linker.TEST_STR_IDS)
    assert f"Data {test_base_linker.TEST_STR_IDS[0]} not found in RedisLinker" in str(record[0].message)
