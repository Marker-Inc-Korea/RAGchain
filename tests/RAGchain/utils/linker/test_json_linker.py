import pytest

from RAGchain.utils.linker import JsonLinker
from RAGchain.utils.linker.base import NoIdWarning, NoDataWarning
import test_base_linker


@pytest.fixture
def json_linker():
    json_linker = JsonLinker(allow_multiple_instances=True)
    yield json_linker
    json_linker.flush_db()


def test_get_json_UUID_UUID(json_linker):
    json_linker.put_json(test_base_linker.TEST_UUID_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    assert json_linker.get_json(test_base_linker.TEST_UUID_IDS) == [test_base_linker.TEST_DB_ORIGIN]


def test_get_json_UUID_STR(json_linker):
    json_linker.put_json(test_base_linker.TEST_UUID_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    assert json_linker.get_json(test_base_linker.TEST_STR_IDS) == [test_base_linker.TEST_DB_ORIGIN]


def test_get_json_STR_UUID(json_linker):
    json_linker.put_json(test_base_linker.TEST_STR_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    assert json_linker.get_json(test_base_linker.TEST_UUID_IDS) == [test_base_linker.TEST_DB_ORIGIN]


def test_get_json_STR_STR(json_linker):
    json_linker.put_json(test_base_linker.TEST_STR_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    assert json_linker.get_json(test_base_linker.TEST_STR_IDS) == [test_base_linker.TEST_DB_ORIGIN]


def test_no_id_warning(json_linker):
    with pytest.warns(NoIdWarning) as record:
        json_linker.get_json(['fake_id'])
    assert "ID fake_id not found in JsonLinker" in str(record[0].message)


def test_no_data_warning(json_linker):
    json_linker.put_json(test_base_linker.TEST_STR_IDS[0], None)
    with pytest.warns(NoDataWarning) as record:
        json_linker.get_json(test_base_linker.TEST_STR_IDS)
    assert f"Data {test_base_linker.TEST_STR_IDS[0]} not found in JsonLinker" in str(record[0].message)
