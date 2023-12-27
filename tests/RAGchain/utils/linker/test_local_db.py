import pytest

from RAGchain.utils.linker import LocalDBSingleton
import test_linker_base


@pytest.fixture
def local_db():
    local_db = LocalDBSingleton(allow_multiple_instances=True)
    local_db.put_json(test_linker_base.TEST_IDS[0], test_linker_base.TEST_DB_ORIGIN)
    yield local_db
    local_db.flush_db()


def test_get_json(local_db):
    assert local_db.get_json(test_linker_base.TEST_IDS) == [test_linker_base.TEST_DB_ORIGIN]
