import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def load_env():
    load_dotenv(dotenv_path='./.env')
    valid_env_keys = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENV', 'MONGO_URL',
                      'MONGO_DB_NAME', 'MONGO_COLLECTION_NAME', 'REDIS_HOST',
                      'REDIS_PORT', 'REDIS_PW', 'HWP_CONVERTER_HOST']
    for key in valid_env_keys:
        assert bool(os.getenv(key)) is True


@pytest.fixture(autouse=True, scope="session")
def test_options():
    make_missing_dir()

    assert TestOptions.root_dir.endswith("tests")
    assert TestOptions.resource_dir.endswith("tests/resources")
    assert os.path.exists(TestOptions.resource_dir)
    assert os.path.exists(TestOptions.chroma_persist_dir)
    assert os.path.exists(TestOptions.source_dir)

    assert not os.path.exists(TestOptions.bm25_db_path), "bm25_db.pkl did not delete properly"
    assert not os.path.exists(TestOptions.pickle_db_path), "pickle_db.pkl did not delete properly"
    assert len(os.listdir(TestOptions.chroma_persist_dir)) == 0, "chroma did not delete properly"


class TestOptions(object):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    resource_dir = os.path.join(root_dir, "resources")
    bm25_db_path = os.path.join(resource_dir, "bm25", "bm25_db.pkl")
    pickle_db_path = os.path.join(resource_dir, "pickle", "pickle_db.pkl")
    chroma_persist_dir = os.path.join(resource_dir, "chroma")
    source_dir = os.path.join(resource_dir, "ingest_files")

    pinecone_namespace = "pinecone-namespace"
    pinecone_index_name = "openai"
    pinecone_dimension = 1536


def make_missing_dir():
    root_dir = TestOptions.resource_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for dir_name in ['bm25', 'chroma', 'ingest_files', 'pickle']:
        if not os.path.exists(os.path.join(root_dir, dir_name)):
            os.makedirs(os.path.join(root_dir, dir_name))
