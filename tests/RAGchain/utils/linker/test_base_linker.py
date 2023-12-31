import pytest
from uuid import uuid4

from RAGchain.utils.linker import DynamoLinker, RedisLinker, SingletonCreationError

TEST_IDS = [uuid4()]

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
