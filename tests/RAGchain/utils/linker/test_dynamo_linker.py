import pytest

from RAGchain.utils.linker import DynamoLinker
from RAGchain.utils.linker.base import NoIdWarning, NoDataWarning
import test_base_linker


@pytest.fixture
def dynamo_db():
    dynamo_db = DynamoLinker(allow_multiple_instances=True)
    dynamo_db.put_json(test_base_linker.TEST_UUID_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    yield dynamo_db
    dynamo_db.flush_db()


def test_get_json(dynamo_db):
    assert dynamo_db.get_json(test_base_linker.TEST_UUID_IDS) == [test_base_linker.TEST_DB_ORIGIN]


def test_no_id_warning(dynamo_db):
    with pytest.warns(NoIdWarning) as record:
        dynamo_db.get_json(['fake_id'])
    assert "ID fake_id not found in DynamoLinker" in str(record[0].message)


def test_no_data_warning(dynamo_db):
    dynamo_db.put_json(test_base_linker.TEST_STR_IDS[0], None)
    with pytest.warns(NoDataWarning) as record:
        dynamo_db.get_json(test_base_linker.TEST_STR_IDS)
    assert f"Data {test_base_linker.TEST_STR_IDS[0]} not found in DynamoLinker" in str(record[0].message)
