import pytest

from RAGchain.utils.linker import RedisLinker, NoIdWarning, NoDataWarning

import test_base_linker

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
    assert redis_db.get_json(TEST_STR_IDS) == [None]
    redis_db.put_json(TEST_UUID_IDS, TEST_DB_ORIGIN)
    with pytest.warns(NoIdWarning) as record:
        assert redis_db.get_json([TEST_UUID_IDS[0],'fake_id']) == [TEST_DB_ORIGIN[0], None]
    assert "ID fake_id not found in Linker" in str(record[0].message)


def test_no_data_warning(redis_db):
    redis_db.put_json(TEST_STR_IDS, [None])
    with pytest.warns(NoDataWarning) as record:
        redis_db.get_json(TEST_STR_IDS)
    assert f"Data {test_base_linker.TEST_STR_IDS[0]} not found in Linker" in str(record[0].message)


def test_no_data_warning2(redis_db):
    full_ids = [TEST_UUID_IDS[0], TEST_STR_IDS[0], TEST_UUID_STR_IDS[0]]
    full_db_origins = [TEST_DB_ORIGIN[0], None, TEST_DB_ORIGIN[0]]
    redis_db.put_json(full_ids, full_db_origins)
    with pytest.warns(NoDataWarning) as record:
        assert redis_db.get_json(full_ids) == [TEST_DB_ORIGIN[0], None, TEST_DB_ORIGIN[0]]
    assert f"Data {TEST_STR_IDS[0]} not found in Linker" in str(record[0].message)
