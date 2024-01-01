import pytest
from uuid import uuid4

from RAGchain.utils.linker import DynamoLinker, RedisLinker, SingletonCreationError, NoIdWarning, NoDataWarning

TEST_UUID_IDS = [uuid4()]
TEST_UUID_STR_IDS = [str(TEST_UUID_IDS[0])]
TEST_STR_IDS = ['test_id']

TEST_DB_ORIGIN = {
    'db_type': 'test_db',
    'db_path': {
        'url': 'test_host',
        'db_name': 'test_port',
        'collection_name': 'test_db_name',
    }
}


def test_singleton_same_child():
    with pytest.raises(SingletonCreationError) as e:
        test_linker_dynamo1 = DynamoLinker()
        test_linker_dynamo2 = DynamoLinker()
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)


def test_allow_multiple_instances():
    test_linker_dynamo1 = DynamoLinker(allow_multiple_instances=True)
    test_linker_dynamo2 = DynamoLinker(allow_multiple_instances=True)
    assert test_linker_dynamo1 is not test_linker_dynamo2


def test_singleton_different_child():
    with pytest.raises(SingletonCreationError) as e:
        test_linker_dynamo = DynamoLinker()
        test_linker_redis = RedisLinker()
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)


def get_json_test(linker, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS):
    # 1. Input: UUID
    linker.put_json(TEST_UUID_IDS[0], TEST_DB_ORIGIN)
    # 1-1. Search: UUID -> Success
    assert linker.get_json(TEST_UUID_IDS) == [TEST_DB_ORIGIN]
    # 1-2. Search: str(UUID) -> Success
    assert linker.get_json(TEST_UUID_STR_IDS) == [TEST_DB_ORIGIN]
    # 1-3. Search: str -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_STR_IDS)
    assert f"ID {TEST_STR_IDS[0]} not found in JsonLinker" in str(record[0].message)
    linker.delete_json(TEST_UUID_IDS[0])

    # 2. Input: str(UUID)
    linker.put_json(TEST_UUID_STR_IDS[0], TEST_DB_ORIGIN)
    # 2-1. Search: UUID -> Success
    assert linker.get_json(TEST_UUID_IDS) == [TEST_DB_ORIGIN]
    # 2-2. Search: str(UUID) -> Success
    assert linker.get_json(TEST_UUID_STR_IDS) == [TEST_DB_ORIGIN]
    # 2-3. Search: str -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_STR_IDS)
    assert f"ID {TEST_STR_IDS[0]} not found in JsonLinker" in str(record[0].message)
    linker.delete_json(TEST_UUID_STR_IDS[0])

    # 3. Input: str
    linker.put_json(TEST_STR_IDS[0], TEST_DB_ORIGIN)
    # 3-1. Search: UUID -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_UUID_IDS)
    assert f"ID {TEST_UUID_IDS[0]} not found in JsonLinker" in str(record[0].message)
    # 3-2. Search: str(UUID) -> Fail
    with pytest.warns(NoIdWarning) as record:
        linker.get_json(TEST_UUID_STR_IDS)
    assert f"ID {TEST_UUID_STR_IDS[0]} not found in JsonLinker" in str(record[0].message)
    # 3-3. Search: str -> Success
    assert linker.get_json(TEST_STR_IDS) == [TEST_DB_ORIGIN]
