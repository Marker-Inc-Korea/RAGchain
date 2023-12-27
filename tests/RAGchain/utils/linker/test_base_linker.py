import pytest

from RAGchain.utils.linker import DynamoDBSingleton, RedisDBSingleton, SingletonCreationError

TEST_IDS = ['test_id_1']

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
        test_linker_dynamo1 = DynamoDBSingleton()
        test_linker_dynamo2 = DynamoDBSingleton()
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)


def test_allow_multiple_instances():
    test_linker_dynamo1 = DynamoDBSingleton(allow_multiple_instances=True)
    test_linker_dynamo2 = DynamoDBSingleton(allow_multiple_instances=True)
    assert test_linker_dynamo1 is not test_linker_dynamo2


def test_singleton_different_child():
    with pytest.raises(SingletonCreationError) as e:
        test_linker_dynamo = DynamoDBSingleton()
        test_linker_redis = RedisDBSingleton()
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)
