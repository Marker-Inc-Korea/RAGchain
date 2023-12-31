import pytest

from RAGchain.utils.linker import JsonLinker
import test_base_linker


@pytest.fixture()
def json_linker():
    json_linker = JsonLinker(allow_multiple_instances=True)
    json_linker.put_json(test_base_linker.TEST_IDS[0], test_base_linker.TEST_DB_ORIGIN)
    yield json_linker
    json_linker.flush_db()


def test_get_json(json_linker):
    assert json_linker.get_json(test_base_linker.TEST_IDS) == [test_base_linker.TEST_DB_ORIGIN]
