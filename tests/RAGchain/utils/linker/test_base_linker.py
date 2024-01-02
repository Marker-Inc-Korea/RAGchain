import warnings
from uuid import uuid4

import pytest

from RAGchain.utils.linker import DynamoLinker, RedisLinker, SingletonCreationError, NoIdWarning, NoDataWarning

TEST_UUID_IDS = [uuid4()]
TEST_UUID_STR_IDS = [str(TEST_UUID_IDS[0])]
TEST_STR_IDS = ['test_id']

TEST_DB_ORIGIN = [{
    'db_type': 'test_db',
    'db_path': {
        'url': 'test_host',
        'db_name': 'test_port',
        'collection_name': 'test_db_name',
    }
}, {
    'db_type': 'mongo_db',
    'db_path': {
        'url': 'test_mongo',
        'db_name': 'test_mongo_port',
        'collection_name': 'test_mongo_name',
    }
}, {
    'db_type': 'sqlite',
    'db_path': {
        'url': 'test_sqlite',
        'db_name': 'test_sqlite_port',
        'collection_name': 'test_sqlite_name',
    }
}]

LONG_TEST_IDS = [uuid4(), uuid4(), str(uuid4()), 'test-1', 'test-2', uuid4(), str(uuid4()), 'test-3']
LONG_DB_ORIGIN = [TEST_DB_ORIGIN[0], TEST_DB_ORIGIN[1], TEST_DB_ORIGIN[2], TEST_DB_ORIGIN[0], TEST_DB_ORIGIN[1],
                  TEST_DB_ORIGIN[2], None, TEST_DB_ORIGIN[1]]


def test_singleton_same_child():
    with pytest.raises(SingletonCreationError) as e:
        test_linker_dynamo1 = DynamoLinker()
        test_linker_dynamo2 = DynamoLinker()
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)


def test_singleton_different_child():
    with pytest.raises(SingletonCreationError) as e:
        test_linker_dynamo = DynamoLinker()
        test_linker_redis = RedisLinker()
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)


def test_allow_multiple_instances():
    test_linker_dynamo1 = DynamoLinker(allow_multiple_instances=True)
    test_linker_dynamo2 = DynamoLinker(allow_multiple_instances=True)
    assert test_linker_dynamo1 is not test_linker_dynamo2


def long_test(linker):
    linker.put_json(LONG_TEST_IDS, LONG_DB_ORIGIN)
    assert linker.get_json(LONG_TEST_IDS) == LONG_DB_ORIGIN
    assert linker.get_json(LONG_TEST_IDS[:3]) == LONG_DB_ORIGIN[:3]
    assert linker.get_json([LONG_TEST_IDS[0], LONG_TEST_IDS[2], LONG_TEST_IDS[4]]) == [LONG_DB_ORIGIN[0],
                                                                                       LONG_DB_ORIGIN[2],
                                                                                       LONG_DB_ORIGIN[4]]
    assert linker.get_json([LONG_TEST_IDS[3], LONG_TEST_IDS[5], LONG_TEST_IDS[7]]) == [LONG_DB_ORIGIN[3],
                                                                                       LONG_DB_ORIGIN[5],
                                                                                       LONG_DB_ORIGIN[7]]
    with pytest.warns(NoDataWarning) as record:
        assert linker.get_json(LONG_TEST_IDS[5:]) == LONG_DB_ORIGIN[5:]
    assert f"Data {LONG_TEST_IDS[6]} not found in Linker" in str(record[0].message)

    linker.delete_json([LONG_TEST_IDS[1], LONG_TEST_IDS[3]])
    with pytest.warns(NoIdWarning) as record:
        assert linker.get_json(LONG_TEST_IDS[1:3]) == [None, LONG_DB_ORIGIN[2]]
    with pytest.warns(NoIdWarning) as record:
        assert linker.get_json([LONG_TEST_IDS[1], LONG_TEST_IDS[3]]) == [None, None]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert linker.get_json([LONG_TEST_IDS[1], LONG_TEST_IDS[6], LONG_TEST_IDS[7]]) == [None, None,
                                                                                           LONG_DB_ORIGIN[7]]
        assert len(w) == 2
        assert issubclass(w[-1].category, NoDataWarning)
        assert issubclass(w[-2].category, NoIdWarning)


def get_json_test(linker, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS):
    # 1. Input: UUID
    linker.put_json(TEST_UUID_IDS, [TEST_DB_ORIGIN[0]])
    # 1-1. Search: UUID -> Success
    assert linker.get_json(TEST_UUID_IDS) == [TEST_DB_ORIGIN[0]]
    # 1-2. Search: str(UUID) -> Success
    assert linker.get_json(TEST_UUID_STR_IDS) == [TEST_DB_ORIGIN[0]]
    # 1-3. Search: str -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_STR_IDS)
    assert f"ID {TEST_STR_IDS[0]} not found in Linker" in str(record[0].message)
    linker.delete_json([TEST_UUID_IDS[0]])

    # 2. Input: str(UUID)
    linker.put_json(TEST_UUID_STR_IDS, [TEST_DB_ORIGIN[0]])
    # 2-1. Search: UUID -> Success
    assert linker.get_json(TEST_UUID_IDS) == [TEST_DB_ORIGIN[0]]
    # 2-2. Search: str(UUID) -> Success
    assert linker.get_json(TEST_UUID_STR_IDS) == [TEST_DB_ORIGIN[0]]
    # 2-3. Search: str -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_STR_IDS)
    assert f"ID {TEST_STR_IDS[0]} not found in Linker" in str(record[0].message)
    linker.delete_json([TEST_UUID_STR_IDS[0]])

    # 3. Input: str
    linker.put_json(TEST_STR_IDS, [TEST_DB_ORIGIN[0]])
    # 3-1. Search: UUID -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_UUID_IDS)
    assert f"ID {TEST_UUID_IDS[0]} not found in Linker" in str(record[0].message)
    # 3-2. Search: str(UUID) -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_UUID_STR_IDS)
    assert f"ID {TEST_UUID_STR_IDS[0]} not found in Linker" in str(record[0].message)
    # 3-3. Search: str -> Success
    assert linker.get_json(TEST_STR_IDS) == [TEST_DB_ORIGIN[0]]


def no_id_warning_test(linker):
    assert linker.get_json(TEST_STR_IDS) == [None]
    linker.put_json(TEST_UUID_IDS, [TEST_DB_ORIGIN[0]])
    with pytest.warns(NoIdWarning) as record:
        assert linker.get_json([TEST_UUID_IDS[0], 'fake_id']) == [TEST_DB_ORIGIN[0], None]
    assert "ID fake_id not found in Linker" in str(record[0].message)


def no_data_warning_test(linker):
    linker.put_json(TEST_STR_IDS, [None])
    with pytest.warns(NoDataWarning) as record:
        linker.get_json(TEST_STR_IDS)
    assert f"Data {TEST_STR_IDS[0]} not found in Linker" in str(record[0].message)


def no_data_warning_test2(linker):
    new_id = uuid4()
    full_ids = [TEST_UUID_IDS[0], TEST_STR_IDS[0], new_id]
    full_db_origins = [TEST_DB_ORIGIN[0], None, TEST_DB_ORIGIN[0]]
    linker.put_json(full_ids, full_db_origins)
    with pytest.warns(NoDataWarning) as record:
        assert linker.get_json(full_ids) == [TEST_DB_ORIGIN[0], None, TEST_DB_ORIGIN[0]]
    assert f"Data {TEST_STR_IDS[0]} not found in Linker" in str(record[0].message)


def delete_test(linker):
    test_id_list = ['test_id1', 'test_id2', 'test_id3', 'test_id4']
    db_origin_list = [TEST_DB_ORIGIN[0] for _ in range(len(test_id_list))]
    linker.put_json(test_id_list, db_origin_list)
    original_data = linker.get_json(test_id_list)
    assert original_data == db_origin_list
    # delete 2 ids
    linker.delete_json(['test_id2', 'test_id4'])
    new_data = linker.get_json(test_id_list)
    assert new_data == [db_origin_list[0], None, db_origin_list[2], None]
