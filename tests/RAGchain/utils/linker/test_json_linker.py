import pytest

from RAGchain.utils.linker import JsonLinker, NoIdWarning, NoDataWarning
import test_base_linker

TEST_UUID_IDS = test_base_linker.TEST_UUID_IDS
TEST_UUID_STR_IDS = test_base_linker.TEST_UUID_STR_IDS
TEST_STR_IDS = test_base_linker.TEST_STR_IDS


@pytest.fixture
def json_linker():
    json_linker = JsonLinker(allow_multiple_instances=True)
    yield json_linker
    json_linker.flush_db()


def test_get_json(json_linker):
    test_base_linker.get_json_test(json_linker, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS)


def test_no_id_warning(json_linker):
    with pytest.warns(NoIdWarning) as record:
        json_linker.get_json(['fake_id'])
    assert "ID fake_id not found in JsonLinker" in str(record[0].message)


def test_no_data_warning(json_linker):
    json_linker.put_json(test_base_linker.TEST_STR_IDS[0], None)
    with pytest.warns(NoDataWarning) as record:
        json_linker.get_json(test_base_linker.TEST_STR_IDS)
    assert f"Data {test_base_linker.TEST_STR_IDS[0]} not found in JsonLinker" in str(record[0].message)
