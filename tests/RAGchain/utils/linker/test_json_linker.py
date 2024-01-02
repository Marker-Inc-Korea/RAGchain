import pytest

import test_base_linker
from RAGchain.utils.linker import JsonLinker

TEST_UUID_IDS = test_base_linker.TEST_UUID_IDS
TEST_UUID_STR_IDS = test_base_linker.TEST_UUID_STR_IDS
TEST_STR_IDS = test_base_linker.TEST_STR_IDS
TEST_DB_ORIGIN = test_base_linker.TEST_DB_ORIGIN


@pytest.fixture
def json_linker():
    json_linker = JsonLinker(allow_multiple_instances=True)
    yield json_linker
    json_linker.flush_db()


def test_get_json(json_linker):
    test_base_linker.get_json_test(json_linker, TEST_UUID_IDS, TEST_UUID_STR_IDS, TEST_STR_IDS)


def test_no_id_warning(json_linker):
    test_base_linker.no_id_warning_test(json_linker)


def test_no_data_warning(json_linker):
    test_base_linker.no_data_warning_test(json_linker)


def test_no_data_warning2(json_linker):
    test_base_linker.no_data_warning_test2(json_linker)


def test_delete(json_linker):
    test_base_linker.delete_test(json_linker)
