import pytest

from RAGchain.utils.linker import DynamoDBSingleton, RedisDBSingleton


def test_singleton_same_child():
    with pytest.raises(Exception) as e:
        test_linker_dynamo1 = DynamoDBSingleton()
        test_linker_dynamo2 = DynamoDBSingleton()
    assert e.type is Exception
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)


def test_singleton_different_child():
    with pytest.raises(Exception) as e:
        test_linker_dynamo = DynamoDBSingleton()
        test_linker_redis = DynamoDBSingleton()
    assert e.type is Exception
    assert "Instance of linker already created. Cannot create another linker." in str(e.value)
